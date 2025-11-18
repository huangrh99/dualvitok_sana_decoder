# Copyright 2025 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import PIL

import torch
import torch.nn as nn
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    SiglipImageProcessor,
    SiglipVisionModel,
    T5EncoderModel,
    T5TokenizerFast,
)

from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode, ToTensor

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, SD3IPAdapterMixin, SD3LoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import calculate_shift, retrieve_timesteps

from tokenizer.builder import build_vq_model
from dataset.build import make_transform

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

from einops import rearrange

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def expand_dims_like(x, y):
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    return x


def transformer_expand_x_embedder(transformer, add_in_channels):
    # expand input channels for transformer to accommodate dualvitok features
    orig_x_embedder = transformer.x_embedder
    new_in_channels = orig_x_embedder.in_features + add_in_channels

    # Create new linear layer with expanded input channels
    new_x_embedder = nn.Linear(
        new_in_channels,
        orig_x_embedder.out_features,
        bias=(orig_x_embedder.bias is not None)
    )

    # Initialize the new layer
    with torch.no_grad():
        # Zero-initialize all weights
        new_x_embedder.weight.zero_()

        # Copy original weights
        new_x_embedder.weight[:, :orig_x_embedder.in_features] = orig_x_embedder.weight.clone()

        # Copy bias if present
        if orig_x_embedder.bias is not None:
            new_x_embedder.bias.copy_(orig_x_embedder.bias)

    # Replace the original embedder
    transformer.x_embedder = new_x_embedder
    return transformer


class AbstractEmbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._is_trainable = None
        self._ucg_rate = None
        self._input_key = None

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @property
    def ucg_rate(self) -> Union[float, torch.Tensor]:
        return self._ucg_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    @ucg_rate.setter
    def ucg_rate(self, value: Union[float, torch.Tensor]):
        self._ucg_rate = value

    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

    @ucg_rate.deleter
    def ucg_rate(self):
        del self._ucg_rate

    @input_key.deleter
    def input_key(self):
        del self._input_key


class DualViTok2ImageEmbedder(AbstractEmbModel):
    def __init__(
            self,
            vq_config,
            vq_model=None,
            vq_ckpt=None,
            use_vq_ckpt_ema=None,
            device="cuda",
            freeze=True,
            image_size=0,
            resize_factor=1,
            not_bicubic=True,
            return_sequence=False,
            grid_feature_scale=1,
            texture_drop_prob=0,
            semantic_drop_prob=0,
            pixel_channel=32,
            semantic_channel=32,
    ):
        super().__init__()
        if vq_model is None:
            vq_model = build_vq_model(vq_config.vq_model)

        if vq_ckpt is not None:
            checkpoint = torch.load(vq_ckpt, map_location='cpu')
            if use_vq_ckpt_ema:
                model_state = checkpoint["ema"]
            else:
                if "model" in checkpoint:
                    model_state = checkpoint["model"]
                else:
                    model_state = checkpoint
            msg = vq_model.load_state_dict(model_state, strict=True)
            print(f"Loaded model from checkpoint: {vq_ckpt} MSG: {msg}")

        # use float16 for efficiency
        vq_model.to(device=device, dtype=torch.float16)
        vq_model.eval()
        min_pixels = 28 * 28
        max_pixels = 512 * 512

        vq_config.data_args.val.augment.update(min_pixels=min_pixels, max_pixels=max_pixels)
        self.processor = make_transform(
            n_px=vq_config.data_args.val.resolution,
            augment=vq_config.data_args.val.augment).transforms[-1]

        self.model = vq_model
        self.device = device
        if freeze:
            self.freeze()

        if image_size > 0:
            preprocessor = [
                Resize(image_size) if not_bicubic else Resize(image_size, interpolation=InterpolationMode.BICUBIC)]
            preprocessor += [
                CenterCrop(image_size),
            ]
            self.preprocessor = Compose(preprocessor)
        self.image_size = image_size
        self.resize_factor = resize_factor
        self.not_bicubic = not_bicubic
        self.return_sequence = return_sequence
        self.grid_feature_scale = grid_feature_scale
        self.texture_drop_prob = texture_drop_prob
        self.semantic_drop_prob = semantic_drop_prob
        self.pixel_channel = pixel_channel
        self.semantic_channel = semantic_channel

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def vq_encode(self, image):
        if isinstance(image, PIL.Image.Image):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = self.processor(image)['image']

        if image.ndim == 5:
            assert image.size(1) == 1
            image = image.squeeze(1)
        bs, _, h, w = image.shape

        if self.image_size > 0:
            image = self.preprocessor(image)
        else:
            if self.resize_factor > 1:
                preprocessor = Resize(
                    (int(h * self.resize_factor), int(w * self.resize_factor))) if self.not_bicubic else \
                    Resize((int(h * self.resize_factor), int(w * self.resize_factor)),
                           interpolation=InterpolationMode.BICUBIC)
                image = preprocessor(image)

        inputs = dict(image=image)
        inputs = self.model.get_input(inputs)

        with torch.no_grad():
            (quant_semantic, diff_semantic, indices_semantic, target_semantic), \
                (quant_detail, diff_detail, indices_detail) = self.model.encode(**inputs)
        return quant_semantic, quant_detail

    def vq_encode_code(self, image):
        if isinstance(image, PIL.Image.Image):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = self.processor(image)['image']

        if image.ndim == 5:
            assert image.size(1) == 1
            image = image.squeeze(1)
        bs, _, h, w = image.shape

        if self.image_size > 0:
            image = self.preprocessor(image)
        else:
            if self.resize_factor > 1:
                preprocessor = Resize(
                    (int(h * self.resize_factor), int(w * self.resize_factor))) if self.not_bicubic else \
                    Resize((int(h * self.resize_factor), int(w * self.resize_factor)),
                           interpolation=InterpolationMode.BICUBIC)
                image = preprocessor(image)

        inputs = dict(image=image)
        inputs = self.model.get_input(inputs)
        with torch.no_grad():
            (quant_semantic, diff_semantic, indices_semantic, target_semantic), \
                (quant_detail, diff_detail, indices_detail) = self.model.encode(**inputs)
        return indices_semantic, indices_detail

    def vq_decode(self, quant_semantic, quant_detail):
        return self.model.decode(quant_semantic, quant_detail)

    def vq_decode_code(self, indices_semantic, indices_detail):
        return self.model.decode_code(indices_semantic, indices_detail)

    @torch.cuda.amp.autocast(dtype=torch.float32)
    def forward(self, image, return_indices=False):
        if isinstance(image, PIL.Image.Image):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = self.processor(image)['image']

        if image.ndim == 5:
            assert image.size(1) == 1
            image = image.squeeze(1)
        bs, _, h, w = image.shape

        if self.image_size > 0:
            image = self.preprocessor(image)
        else:
            if self.resize_factor > 1:
                preprocessor = Resize(
                    (int(h * self.resize_factor), int(w * self.resize_factor))) if self.not_bicubic else \
                    Resize((int(h * self.resize_factor), int(w * self.resize_factor)),
                           interpolation=InterpolationMode.BICUBIC)
                image = preprocessor(image)

        inputs = dict(image=image)
        inputs = self.model.get_input(inputs)

        with torch.no_grad():
            (quant_semantic, diff_semantic, indices_semantic, target_semantic), \
                (quant_detail, diff_detail, indices_detail) = self.model.encode(**inputs)

            feature = self.model.merge_quants(quant_semantic, quant_detail)

        if self.training and self.semantic_drop_prob > 0.0:
            feature[:, :self.semantic_channel] = (
                    expand_dims_like(
                        torch.bernoulli(
                            (1.0 - self.semantic_drop_prob)
                            * torch.ones(feature.shape[0], device=feature.device)
                        ),
                        feature[:, :self.semantic_channel],
                    )
                    * feature[:, :self.semantic_channel]
            )
        if self.training and self.texture_drop_prob > 0.0:
            feature[:, self.semantic_channel:] = (
                    expand_dims_like(
                        torch.bernoulli(
                            (1.0 - self.texture_drop_prob)
                            * torch.ones(feature.shape[0], device=feature.device)
                        ),
                        feature[:, self.semantic_channel:],
                    )
                    * feature[:, self.semantic_channel:]
            )

        if self.return_sequence:
            feature = rearrange(feature, 'b c h w -> b h w c')
            _, this_h, this_w, _ = feature.shape
            feature = feature.view(bs, this_w * this_w, -1)
        else:
            feature = feature * self.grid_feature_scale

        if return_indices:
            return feature, indices_semantic, indices_detail

        return feature

    def encode(self, img):
        return self(img)

    def indices_to_codes(self, semantic_indices, texture_indices):
        quant_semantic, quant_texture = self.model.indices_to_codes(semantic_indices, texture_indices)
        return self.model.merge_quants(quant_semantic, quant_texture)


def is_pil_image(obj):
    return isinstance(obj, PIL.Image.Image)


class StableDiffusion3DecoderPipeline(DiffusionPipeline, SD3LoraLoaderMixin, FromSingleFileMixin, SD3IPAdapterMixin):
    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->image_encoder->transformer->vae"
    _optional_components = ["image_encoder", "feature_extractor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "pooled_prompt_embeds"]

    def __init__(
            self,
            transformer: SD3Transformer2DModel,
            scheduler: FlowMatchEulerDiscreteScheduler,
            vae: AutoencoderKL,
            image_encoder: SiglipVisionModel,
            feature_extractor: SiglipImageProcessor,
            vq_config=None,
            vq_model=None,
            vq_model_embedder=None,
            vq_embed_projector=None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )
        self.patch_size = (
            self.transformer.config.patch_size if hasattr(self, "transformer") and self.transformer is not None else 2
        )

        # vq embedder.
        if vq_model_embedder is not None:
            self.vq_model_embedder = vq_model_embedder
            self.dualvitok_channels = self.vq_model_embedder.model.pixel_channel + self.vq_model_embedder.model.semantic_channel

        elif vq_config is not None:
            # Initialize DualViTok embedder if vq_config and vq_model are provided

            embedder_kwargs = dict(image_size=0,
                                   resize_factor=1,
                                   return_sequence=False,
                                   grid_feature_scale=0.7)

            self.vq_model_embedder = DualViTok2ImageEmbedder(vq_config, vq_model, **embedder_kwargs)

            self.dualvitok_channels = self.vq_model_embedder.model.pixel_channel + self.vq_model_embedder.model.semantic_channel
        self.vq_embed_projector = vq_embed_projector

        self.empty_prompt_embeds = None
        self.empty_pooled_prompt_embeds = None

    def set_prompt_embeds(self, empty_prompt_embeds, empty_pooled_prompt_embeds):
        self.empty_prompt_embeds = empty_prompt_embeds
        self.empty_pooled_prompt_embeds = empty_pooled_prompt_embeds

    def _get_t5_prompt_embeds(
            self,
            prompt: Union[str, List[str]] = None,
            num_images_per_prompt: int = 1,
            max_sequence_length: int = 256,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.text_encoder_3 is None:
            return torch.zeros(
                (
                    batch_size * num_images_per_prompt,
                    max_sequence_length,
                    self.transformer.config.joint_attention_dim,
                ),
                device=device,
                dtype=dtype,
            )

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1: -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]

        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
            self,
            prompt: Union[str, List[str]],
            num_images_per_prompt: int = 1,
            device: Optional[torch.device] = None,
            clip_skip: Optional[int] = None,
            clip_model_index: int = 0,
    ):
        device = device or self._execution_device

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1: -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds

    def encode_prompt(
            self,
            prompt: Union[str, List[str]],
            prompt_2: Union[str, List[str]],
            prompt_3: Union[str, List[str]],
            device: Optional[torch.device] = None,
            num_images_per_prompt: int = 1,
            do_classifier_free_guidance: bool = True,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            negative_prompt_3: Optional[Union[str, List[str]]] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            clip_skip: Optional[int] = None,
            max_sequence_length: int = 256,
            lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            negative_prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used in all the text-encoders.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, SD3LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=0,
            )
            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                prompt=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=1,
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

            t5_prompt_embed = self._get_t5_prompt_embeds(
                prompt=prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
            )

            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_3 = negative_prompt_3 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )
            negative_prompt_3 = (
                batch_size * [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embed, negative_pooled_prompt_embed = self._get_clip_prompt_embeds(
                negative_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=0,
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                negative_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=1,
            )
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

            t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                prompt=negative_prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1]),
            )

            negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
            )

        if self.text_encoder is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def check_inputs(
            self,
            height,
            width,
            callback_on_step_end_tensor_inputs=None,
            max_sequence_length=None,
    ):
        if (
                height % (self.vae_scale_factor * self.patch_size) != 0
                or width % (self.vae_scale_factor * self.patch_size) != 0
        ):
            raise ValueError(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * self.patch_size} but are {height} and {width}."
                f"You can use height {height - height % (self.vae_scale_factor * self.patch_size)} and width {width - width % (self.vae_scale_factor * self.patch_size)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
                k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    def prepare_latents(
            self,
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def skip_guidance_layers(self):
        return self._skip_guidance_layers

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    # Adapted from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.encode_image
    def encode_image(self, image: PipelineImageInput, device: torch.device) -> torch.Tensor:
        """Encodes the given image into a feature representation using a pre-trained image encoder.

        Args:
            image (`PipelineImageInput`):
                Input image to be encoded.
            device: (`torch.device`):
                Torch device.

        Returns:
            `torch.Tensor`: The encoded image feature representation.
        """
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=self.dtype)

        return self.image_encoder(image, output_hidden_states=True).hidden_states[-2]

    # Adapted from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.prepare_ip_adapter_image_embeds
    def prepare_ip_adapter_image_embeds(
            self,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            ip_adapter_image_embeds: Optional[torch.Tensor] = None,
            device: Optional[torch.device] = None,
            num_images_per_prompt: int = 1,
            do_classifier_free_guidance: bool = True,
    ) -> torch.Tensor:
        """Prepares image embeddings for use in the IP-Adapter.

        Either `ip_adapter_image` or `ip_adapter_image_embeds` must be passed.

        Args:
            ip_adapter_image (`PipelineImageInput`, *optional*):
                The input image to extract features from for IP-Adapter.
            ip_adapter_image_embeds (`torch.Tensor`, *optional*):
                Precomputed image embeddings.
            device: (`torch.device`, *optional*):
                Torch device.
            num_images_per_prompt (`int`, defaults to 1):
                Number of images that should be generated per prompt.
            do_classifier_free_guidance (`bool`, defaults to True):
                Whether to use classifier free guidance or not.
        """
        device = device or self._execution_device

        if ip_adapter_image_embeds is not None:
            if do_classifier_free_guidance:
                single_negative_image_embeds, single_image_embeds = ip_adapter_image_embeds.chunk(2)
            else:
                single_image_embeds = ip_adapter_image_embeds
        elif ip_adapter_image is not None:
            single_image_embeds = self.encode_image(ip_adapter_image, device)
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.zeros_like(single_image_embeds)
        else:
            raise ValueError("Neither `ip_adapter_image_embeds` or `ip_adapter_image_embeds` were provided.")

        image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)

        if do_classifier_free_guidance:
            negative_image_embeds = torch.cat([single_negative_image_embeds] * num_images_per_prompt, dim=0)
            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0)

        return image_embeds.to(device=device)

    def enable_sequential_cpu_offload(self, *args, **kwargs):
        if self.image_encoder is not None and "image_encoder" not in self._exclude_from_cpu_offload:
            logger.warning(
                "`pipe.enable_sequential_cpu_offload()` might fail for `image_encoder` if it uses "
                "`torch.nn.MultiheadAttention`. You can exclude `image_encoder` from CPU offloading by calling "
                "`pipe._exclude_from_cpu_offload.append('image_encoder')` before `pipe.enable_sequential_cpu_offload()`."
            )

        super().enable_sequential_cpu_offload(*args, **kwargs)

    @torch.no_grad()
    def __call__(
            self,
            image,
            vq_indices: Optional[torch.LongTensor] = None,
            vq_embeds: Optional[torch.LongTensor] = None,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            prompt_3: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 28,
            sigmas: Optional[List[float]] = None,
            guidance_scale: float = 7.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            negative_prompt_3: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            ip_adapter_image_embeds: Optional[torch.Tensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            clip_skip: Optional[int] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            max_sequence_length: int = 256,
            skip_guidance_layers: List[int] = None,
            skip_layer_guidance_scale: float = 2.8,
            skip_layer_guidance_stop: float = 0.2,
            skip_layer_guidance_start: float = 0.01,
            mu: Optional[float] = None,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            height,
            width,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._skip_layer_guidance_scale = skip_layer_guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if image is not None and isinstance(image, str):
            batch_size = 1
            image = [Image.open(image)]
        elif image is not None and isinstance(image, PIL.Image.Image):
            batch_size = 1
            image = [image]
        elif image is not None and isinstance(image, list):
            batch_size = len(image)
            if not is_pil_image(image[0]):
                from PIL import Image
                image = [Image.open(item) for item in image]
        elif image is not None and is_pil_image(image):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        # (
        #     prompt_embeds,
        #     negative_prompt_embeds,
        #     pooled_prompt_embeds,
        #     negative_pooled_prompt_embeds,
        # ) = self.encode_prompt(
        #     prompt=prompt,
        #     prompt_2=prompt_2,
        #     prompt_3=prompt_3,
        #     negative_prompt=negative_prompt,
        #     negative_prompt_2=negative_prompt_2,
        #     negative_prompt_3=negative_prompt_3,
        #     do_classifier_free_guidance=self.do_classifier_free_guidance,
        #     prompt_embeds=prompt_embeds,
        #     negative_prompt_embeds=negative_prompt_embeds,
        #     pooled_prompt_embeds=pooled_prompt_embeds,
        #     negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        #     device=device,
        #     clip_skip=self.clip_skip,
        #     num_images_per_prompt=num_images_per_prompt,
        #     max_sequence_length=max_sequence_length,
        #     lora_scale=lora_scale,
        # )
        # 3. Process image embeddings if DualViTok is available
        if vq_indices or vq_embeds:
            if vq_indices:
                indices_semantic, indices_detail = vq_indices[0], vq_indices[1]
                vq_embeds = self.vq_model_embedder.indices_to_codes(
                    indices_semantic, indices_detail)
        elif image is not None:
            vq_embeds, indices_semantic, indices_detail = self.vq_model_embedder(image, return_indices=True)

        if self.do_classifier_free_guidance:
            negative_vq_embeds = torch.zeros_like(vq_embeds)
            vq_embeds = torch.cat([negative_vq_embeds, vq_embeds], dim=0)

        # if self.do_classifier_free_guidance:
        #     if skip_guidance_layers is not None:
        #         original_prompt_embeds = prompt_embeds
        #         original_pooled_prompt_embeds = pooled_prompt_embeds
        #     prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        #     pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        # 3.5 Get prompt embeds
        if self.vq_embed_projector is not None:
            prompt_embeds = rearrange(self.vq_embed_projector(vq_embeds), 'b c h w -> b (h w) c')
            pooled_prompt_embeds = prompt_embeds.mean(1)
            print('using vq_embedder projector')
        else:
            prompt_embeds = self.empty_prompt_embeds.repeat(vq_embeds.shape[0], 1, 1)
            pooled_prompt_embeds = self.empty_pooled_prompt_embeds.repeat(vq_embeds.shape[0], 1)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels - self.dualvitok_channels

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        if latents.shape[-2:] != vq_embeds.shape[-2:]:
            vq_embeds = torch.nn.functional.interpolate(vq_embeds, size=latents.shape[-2:])

        # 5. Prepare timesteps
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            _, _, height, width = latents.shape
            image_seq_len = (height // self.transformer.config.patch_size) * (
                    width // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Prepare image embeddings
        if (ip_adapter_image is not None and self.is_ip_adapter_active) or ip_adapter_image_embeds is not None:
            ip_adapter_image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

            if self.joint_attention_kwargs is None:
                self._joint_attention_kwargs = {"ip_adapter_image_embeds": ip_adapter_image_embeds}
            else:
                self._joint_attention_kwargs.update(ip_adapter_image_embeds=ip_adapter_image_embeds)

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])
                latent_model_input = torch.cat([latent_model_input, vq_embeds], dim=1)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    pooled_prompt_embeds = callback_outputs.pop("pooled_prompt_embeds", pooled_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)
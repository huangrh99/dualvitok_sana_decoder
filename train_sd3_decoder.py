#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import copy
import gc
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast

from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict

import PIL
import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline as StableDiffusion3PipelineOrg,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)

# Add missing imports
from dataset.build import make_transform, build_dataset
from dataset.multi_ratio_dataset import ResolutionBatchSampler
from utils.registry_utils import read_config
from tokenizer.sd3_decoder_pipe import StableDiffusion3DecoderPipeline as StableDiffusion3Pipeline, \
    DualViTok2ImageEmbedder
from utils.dist_utils import get_rank

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
import types
import wandb

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__)


def save_model_card(
        repo_id: str,
        images=None,
        base_model: str = None,
        instance_prompt=None,
        validation_prompt=None,
        repo_folder=None,
):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"image_{i}.png"}}
            )

    model_description = f"""
# SD3 DreamBooth - {repo_id}

<Gallery />

## Model description

These are {repo_id} DreamBooth weights for {base_model}.

The weights were trained  using [DreamBooth](https://dreambooth.github.io/).

## Trigger words

You should use {instance_prompt} to trigger the image generation.

## Download model

[Download]({repo_id}/tree/main) them in the Files & versions tab.

## License

Please adhere to the licensing terms as described `[here](https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/LICENSE)`.
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="openrail++",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "sd3",
        "sd3-diffusers",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def load_text_encoders(class_one, class_two, class_three):
    text_encoder_one = class_one.from_pretrained(
        args.model.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.model.revision,
        variant=args.model.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.model.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.model.revision,
        variant=args.model.variant
    )
    text_encoder_three = class_three.from_pretrained(
        args.model.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.model.revision,
        variant=args.model.variant
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


def log_validation(
        pipeline,
        args,
        accelerator,
        pipeline_args,
        epoch,
        is_final_validation=False,
):
    logger.info(
        f"Running validation..."
        f" {args.val_data.validation_image}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(
        args.training.seed) if args.training.seed else None
    # autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()
    autocast_ctx = nullcontext()

    original_image = Image.open(args.val_data.validation_image).convert('RGB')

    # Resize original image to target resolution
    original_image_resized = original_image.resize((args.val_data.resolution // 2, args.val_data.resolution // 2),
                                                   Image.Resampling.LANCZOS)
    target_size = (args.val_data.resolution, args.val_data.resolution)

    guidance_scales = [0, 1.5, 2, 3]
    with autocast_ctx:
        images = [pipeline(
            image=original_image_resized,
            height=args.val_data.resolution,
            width=args.val_data.resolution,
            guidance_scale=cfg_scale, generator=generator).images[0] for cfg_scale in guidance_scales]

    # Create comparison images

    # Log to trackers
    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            images_to_log = [original_image_resized.resize(target_size)] + images

            # Stack original and reconstructed images
            np_images = np.stack([np.asarray(img) for img in images_to_log])
            tracker.writer.add_images(f"{phase_name}/reconstruction", np_images, epoch, dataformats="NHWC")

    if accelerator.is_main_process and "wandb" in args.logging_config.report_to:
        images_to_log = [
            wandb.Image(original_image_resized.resize(target_size), caption="Original")
        ]
        images_to_log += [
            wandb.Image(img, caption=f"Reconstructed_cfg{cfg}") for img, cfg in zip(images, guidance_scales)
        ]

        wandb.log(
            {
                f"{phase_name}/reconstruction": images_to_log
            }
        )

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images


def import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


@torch.no_grad
def compute_vae_encodings(batch, vae, dualvitok, diffusion_upsample_rate=1):
    images = batch.pop("pixel_values")
    if images.ndim == 5:
        images = images.squeeze(0)
    pixel_values = torch.stack(list(images))
    pixel_values = images.to(memory_format=torch.contiguous_format, device=vae.device, dtype=vae.dtype)

    model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor
    b, c, h, w = pixel_values.shape
    if diffusion_upsample_rate > 1:
        h2, w2 = h // diffusion_upsample_rate, w // diffusion_upsample_rate
        pixel_values = torch.nn.functional.interpolate(pixel_values, size=(h2, w2), mode='bicubic')

    model_input2 = dualvitok(pixel_values)

    if model_input2.shape[-2:] != model_input.shape[-2:]:
        model_input2 = torch.nn.functional.interpolate(model_input2, size=model_input.shape[-2:])

    batch.update({"model_input": model_input.detach(), "model_input2": model_input2.detach()})
    return batch


def transformer_expand_input_channels(transformer, add_in_channels):
    """
    Expand transformer input channels to accommodate additional VQ-GAN features.
    Similar to unet_expand_conv_in but for SD3 transformer.
    """
    # For SD3 transformer, we need to modify the input projection layer
    # This is typically the first linear layer that processes the input latents
    orig_proj = transformer.pos_embed.proj
    new_in_channels = orig_proj.in_channels + add_in_channels

    new_proj = torch.nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=orig_proj.out_channels,
        kernel_size=orig_proj.kernel_size,
        stride=orig_proj.stride,
        padding=orig_proj.padding,
        bias=(orig_proj.bias is not None)
    )

    # Copy original weights
    with torch.no_grad():
        new_proj.weight.zero_()
        new_proj.weight[:, :orig_proj.in_channels] = orig_proj.weight
        # Initialize new channels with small random values
        # torch.nn.init.kaiming_normal_(new_proj.weight[:, orig_proj.in_channels:], mode='fan_out')
        if new_proj.bias is not None and orig_proj.bias is not None:
            new_proj.bias.copy_(orig_proj.bias)

    transformer.pos_embed.proj = new_proj

    return transformer


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    input_args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(input_args.local_rank)
    args = read_config(input_args.config)
    args.local_rank = input_args.local_rank

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
        text_encoder,
        tokenizer,
        prompt=None,
        num_images_per_prompt=1,
        device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
        text_encoder,
        tokenizer,
        prompt: str,
        device=None,
        num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


def encode_prompt(
        text_encoders,
        tokenizers,
        prompt: str,
        device=None,
        num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds


def collate_fn(examples):
    # Handle both tuple format (from batch sampler) and dict format
    if isinstance(examples[0], tuple) and isinstance(examples[0][1], int):
        examples = [example[0] for example in examples]

    pixel_values = [example["pixel_values"] for example in examples]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values}

    # Add additional fields if they exist (for multi-resolution datasets)
    if "original_sizes" in examples[0]:
        batch["original_sizes"] = [example["original_sizes"] for example in examples]
    if "crop_top_lefts" in examples[0]:
        batch["crop_top_lefts"] = [example["crop_top_lefts"] for example in examples]

    return batch


def main(args):
    if args.logging_config.report_to == "wandb" and args.logging_config.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.logging_config.output_dir, args.logging_config.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.logging_config.output_dir,
                                                      logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.training.gradient_accumulation_steps,
        mixed_precision=args.training.mixed_precision,
        log_with=args.logging_config.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.logging_config.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.training.seed is not None:
        set_seed(args.training.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.logging_config.output_dir is not None:
            os.makedirs(args.logging_config.output_dir, exist_ok=True)

        if args.logging_config.push_to_hub:
            repo_id = create_repo(
                repo_id=args.logging_config.hub_model_id or Path(args.logging_config.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.model.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.model.revision,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.model.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.model.revision,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.model.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.model.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.model.pretrained_model_name_or_path, args.model.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.model.pretrained_model_name_or_path, args.model.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.model.pretrained_model_name_or_path, args.model.revision, subfolder="text_encoder_3"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
    )
    vae = AutoencoderKL.from_pretrained(
        args.model.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.model.revision,
        variant=args.model.variant,
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        args.model.pretrained_model_name_or_path, subfolder="transformer", revision=args.model.revision,
        variant=args.model.variant
    )

    # Load dualvitok config and create embedder
    vq_config = read_config(args.model.dualvitok_config)
    vq_model_embedder = DualViTok2ImageEmbedder(vq_config, vq_ckpt=args.model.dualvitok_ckpt)
    # Expand transformer input channels to accommodate DualViTok features
    additional_channels = vq_model_embedder.pixel_channel + vq_model_embedder.semantic_channel
    print(f'Expanding transformer input channels by {additional_channels} for DualViTok features')
    transformer = transformer_expand_input_channels(transformer, additional_channels)

    # Update transformer config
    if hasattr(transformer.config, 'in_channels'):
        transformer.register_to_config(in_channels=transformer.config.in_channels + additional_channels)

    vq_embed_projector = None
    # if args.model.use_vq_embed_projector:
    #     vq_embed_projector = GLUMBConv(
    #             in_channels=additional_channels,
    #             out_channels=2240, # empty_prompt_embeds.shape[-1],
    #             expand_ratio=4,
    #             norm_type='rms_norm',
    #             residual_connection=False,
    #         ).to(weight_dtype)

    transformer.requires_grad_(True)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=torch.float32)

    if args.training.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.lora.lora_pretrained_path:
        if args.lora.lora_layers is not None:
            target_modules = [layer.strip() for layer in args.lora.lora_layers.split(",")]
        else:
            target_modules = ["to_k", "to_q", "to_v", "to_out.0",
                              "conv_point",
                              #   "conv_depth",
                              "conv_inverted"
                              ]

        # now we will add new LoRA weights the transformer layers
        transformer_lora_config = LoraConfig(
            r=args.lora.rank,
            lora_alpha=args.lora.rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        transformer.add_adapter(transformer_lora_config)

        lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(args.lora.lora_pretrained_path)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name="default")

        pipeline = StableDiffusion3Pipeline.from_pretrained(
            args.model.pretrained_model_name_or_path,
            transformer=transformer,
            vae=vae,
            vq_model_embedder=vq_model_embedder,
            empty_prompt_embeds=empty_prompt_embeds,
            empty_pooled_prompt_embeds=empty_pooled_prompt_embeds,
            torch_dtype=torch.float32,
        )

        pipeline.transformer.fuse_lora(adapter_names=["default"], lora_scale=1.0)
        pipeline.unload_lora_weights()
        transformer = pipeline.transformer

        if os.path.exists(os.path.join(args.lora.lora_pretrained_path, "sd3_pos_embed.bin")):
            transformer.pos_embed.load_state_dict(
                torch.load(os.path.join(args.lora.lora_pretrained_path, "sd3_pos_embed.bin")))

        shift_table_path = os.path.join(args.lora.lora_pretrained_path, 'sd3_scale_shift_table.bin')
        if os.path.exists(shift_table_path):
            shift_table_state_dict = torch.load(shift_table_path, 'cpu')
            transformer.load_state_dict(shift_table_state_dict, strict=False)

    if args.lora.use_lora:
        if args.lora.lora_layers is not None:
            target_modules = [layer.strip() for layer in args.lora.lora_layers.split(",")]
        else:
            target_modules = [
                "to_k", "to_q", "to_v", "to_out.0",
                "ff_context.net.0.proj",
                "ff.net.0.proj",
            ]

        # now we will add new LoRA weights the transformer layers
        transformer_lora_config = LoraConfig(
            r=args.lora.rank,
            lora_alpha=args.lora.rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        transformer.add_adapter(transformer_lora_config)
    else:
        transformer.requires_grad_(True)

    if args.training.tune_scale_shift:
        for name, param in transformer.named_parameters():
            if "scale_shift_table" in name:
                param.requires_grad_(True)
    if vq_embed_projector is not None:
        vq_embed_projector.requires_grad_(True)
        for n, p in transformer.named_parameters():
            if "attn3" in n:
                p.requires_grad_(True)
    transformer.pos_embed.requires_grad_(True)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            vq_embed_projector_to_save = None
            scale_shift_table_to_save = None
            pos_embed_to_save = None

            for i, model in enumerate(models):
                if isinstance(unwrap_model(model), SD3Transformer2DModel):
                    if args.lora.use_lora:
                        if args.training.tune_scale_shift:
                            # read requires_grad and remove lora
                            scale_shift_table_to_save = {}
                            for n, p in unwrap_model(transformer).state_dict().items():
                                if 'scale_shift_table' in n:
                                    scale_shift_table_to_save[n] = p
                                torch.save(scale_shift_table_to_save,
                                           os.path.join(output_dir, "sd3_scale_shift_table.bin"))

                        if args.model.use_vq_embed_projector:
                            attn3 = {n: p for n, p in model.state_dict().items() if 'attn3' in n}
                            torch.save(attn3, os.path.join(output_dir, "sd3_attn3.bin"))

                        transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                        StableDiffusion3Pipeline.save_lora_weights(
                            output_dir,
                            transformer_lora_layers=transformer_lora_layers_to_save,
                        )

                        pos_embed_to_save = unwrap_model(transformer).pos_embed.state_dict()
                        torch.save(pos_embed_to_save, os.path.join(output_dir, "sd3_pos_embed.bin"))
                    else:
                        unwrap_model(model).save_pretrained(os.path.join(output_dir, "transformer"))
                elif isinstance(model, type(unwrap_model(vq_embed_projector))):
                    vq_embed_projector_to_save = unwrap_model(vq_embed_projector).state_dict()
                    torch.save(vq_embed_projector_to_save, os.path.join(output_dir, "sd3_vq_embed_projector.bin"))
                elif isinstance(unwrap_model(model), (CLIPTextModelWithProjection, T5EncoderModel)):
                    if isinstance(unwrap_model(model), CLIPTextModelWithProjection):
                        hidden_size = unwrap_model(model).config.hidden_size
                        if hidden_size == 768:
                            unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder"))
                        elif hidden_size == 1280:
                            unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder_2"))
                    else:
                        unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder_3"))
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            if isinstance(unwrap_model(model), SD3Transformer2DModel):
                if args.lora.use_lora:
                    lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(input_dir)

                    transformer_state_dict = {
                        f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if
                        k.startswith("transformer.")
                    }
                    transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
                    incompatible_keys = set_peft_model_state_dict(model, transformer_state_dict, adapter_name="default")
                    if incompatible_keys is not None:
                        # check only for unexpected keys
                        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                        if unexpected_keys:
                            logger.warning(
                                f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                                f" {unexpected_keys}. "
                            )
                    if os.path.exists(os.path.join(input_dir, "sd3_pos_embed.bin")):
                        model.pos_embed.load_state_dict(torch.load(os.path.join(input_dir, "sd3_pos_embed.bin"), 'cpu'))

                    shift_table_path = os.path.join(input_dir, 'sd3_scale_shift_table.bin')
                    if os.path.exists(shift_table_path):
                        model.load_state_dict(torch.load(shift_table_path, 'cpu'), strict=False)
                else:
                    load_model = SD3Transformer2DModel.from_pretrained(input_dir, subfolder="transformer")
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model
            else:
                raise ValueError(f"Unsupported model found: {type(model)=}")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.training.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.training.scale_lr:
        args.training.learning_rate = (
                args.training.learning_rate * args.training.gradient_accumulation_steps * args.data.train_batch_size * accelerator.num_processes
        )

    # Optimization parameters
    params_to_optimize = [p for p in transformer.parameters() if p.requires_grad]
    transformer_parameters_with_lr = {"params": params_to_optimize, "lr": args.training.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.type.lower() == "prodigy" or args.optimizer.type.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer.type}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.training.use_8bit_adam and not args.optimizer.type.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.type.lower()}"
        )

    if args.optimizer.type.lower() == "adamw":
        if args.training.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.optimizer.adam_beta1, args.optimizer.adam_beta2),
            weight_decay=args.optimizer.adam_weight_decay,
            eps=args.optimizer.adam_epsilon,
        )

    if args.optimizer.type.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.training.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.training.learning_rate,
            betas=(args.optimizer.adam_beta1, args.optimizer.adam_beta2),
            beta3=args.optimizer.prodigy_beta3,
            weight_decay=args.optimizer.adam_weight_decay,
            eps=args.optimizer.adam_epsilon,
            decouple=args.optimizer.prodigy_decouple,
            use_bias_correction=args.optimizer.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Dataset and DataLoaders creation:
    if args.data.dataset_name is not None and args.data.dataset_name == 'multires':
        train_dataset = build_dataset(args.data.dataset_args)
        batch_sampler = ResolutionBatchSampler(train_dataset, args.data.train_batch_size, drop_last=True)
    elif args.data.dataset_name is not None and args.data.dataset_name == 'custom':
        train_dataset = build_dataset(args.data.dataset_args)
        batch_sampler = None
    else:
        # Use huggingface datasets as fallback
        from datasets import load_dataset

        # Placeholder for now - you would need to implement proper dataset loading
        # based on your specific requirements
        train_dataset = load_dataset("imagefolder", data_dir="./data", split="train")
        batch_sampler = None

    if batch_sampler:
        dataloader_batch_size = 1
        shuffle = None
    else:
        dataloader_batch_size = args.data.train_batch_size
        shuffle = True

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=dataloader_batch_size,
        batch_sampler=batch_sampler,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=args.data.dataloader_num_workers,
    )

    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]

    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt)
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds

    empty_prompt_embeds, empty_pooled_prompt_embeds = compute_text_embeddings(
        prompt='',  # empty
        text_encoders=text_encoders,
        tokenizers=tokenizers,
    )
    # empty_prompt_embeds torch.Size([1, 154, 4096]) torch.Size([1, 2048])
    print("empty_prompt_embeds", empty_prompt_embeds.shape, empty_pooled_prompt_embeds.shape)
    del text_encoder_one, text_encoder_two, text_encoder_three
    del tokenizer_one, tokenizer_two, tokenizer_three
    gc.collect()
    torch.cuda.empty_cache()

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.training.gradient_accumulation_steps)
    if args.training.max_train_steps is None:
        args.training.max_train_steps = args.training.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.training.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.training.max_train_steps * accelerator.num_processes,
        num_cycles=args.training.lr_num_cycles,
        power=args.training.lr_power,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.training.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.training.max_train_steps = args.training.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.training.num_train_epochs = math.ceil(args.training.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        if "wandb" in args.logging_config.report_to:
            wandb_init_kwargs = {
                "name": args.logging_config.run_name,
                "dir": args.logging_config.run_name,  # Specify wandb directory
                "save_code": True,  # Save code snapshot
                "resume": "allow",  # Allow resuming from checkpoints
            }
            wandb.init(project='bagel-runhui', **wandb_init_kwargs)

        tracker_name = "dualvitok-ds3-decoder"
        accelerator.init_trackers(tracker_name)

    # Train!
    total_batch_size = args.data.train_batch_size * accelerator.num_processes * args.training.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.training.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.data.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.training.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.training.max_train_steps}")
    logger.info(f"  DualViTok channels: {vq_model_embedder.pixel_channel + vq_model_embedder.semantic_channel}")

    total_params = sum(p.numel() for p in transformer.parameters())
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {total_params / 1e6:.2f}M")
    logger.info(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")

    global_step = 0
    first_epoch = 0

    if accelerator.is_main_process:
        for n, p in transformer.named_parameters():
            logger.info(f"{n} {p.dtype} {p.requires_grad}")

    # Potentially load in the weights and states from a previous save
    if args.training.resume_from_checkpoint:
        if args.training.resume_from_checkpoint != "latest":
            path = os.path.basename(args.training.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.logging_config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.training.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.training.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.logging_config.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

        if optimizer.param_groups[0]['lr'] != args.training.learning_rate:
            new_lr = args.training.learning_rate
            print(f'Change learning rate to {new_lr}')
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr

            lr_scheduler.scheduler.base_lrs = [new_lr for _ in lr_scheduler.scheduler.base_lrs]

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.training.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.training.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                prompt_embeds = empty_prompt_embeds.repeat(batch["pixel_values"].shape[0], 1, 1)
                pooled_prompt_embeds = empty_pooled_prompt_embeds.repeat(batch["pixel_values"].shape[0], 1)
                vae = vae.to(accelerator.device)
                batch = compute_vae_encodings(batch, vae, vq_model_embedder,
                                              diffusion_upsample_rate=args.model.diffusion_upsample_rate)

                model_input = batch["model_input"].to(accelerator.device, dtype=weight_dtype)
                model_input2 = batch["model_input2"].to(accelerator.device, dtype=weight_dtype)

                if hasattr(args.training, 'proportion_empty_prompts') and args.training.proportion_empty_prompts:
                    if random.random() < args.training.proportion_empty_prompts:
                        model_input2 = torch.zeros_like(model_input2)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                indices = torch.randint(0, noise_scheduler_copy.config.num_train_timesteps, (bsz,))
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input

                # Predict the noise residual
                transformer_input = torch.cat([noisy_model_input, model_input2], dim=1).contiguous()

                model_pred = transformer(
                    hidden_states=transformer_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]

                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                # Preconditioning of the model outputs.
                model_pred = model_pred * (-sigmas) + noisy_model_input

                # TODO (kashif, sayakpaul): weighting sceme needs to be experimented with :)
                if args.training.weighting_scheme == "sigma_sqrt":
                    weighting = (sigmas ** -2.0).float()
                elif args.training.weighting_scheme == "logit_normal":
                    # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
                    u = torch.normal(mean=args.logit_mean, std=args.logit_std, size=(bsz,), device=accelerator.device)
                    weighting = torch.nn.functional.sigmoid(u)
                elif args.training.weighting_scheme == "mode":
                    # See sec 3.1 in the SD3 paper (20).
                    u = torch.rand(size=(bsz,), device=accelerator.device)
                    weighting = 1 - u - args.mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
                else:
                    weighting = torch.as_tensor(1)
                # simplified flow matching aka 0-rectified flow matching loss
                # target = model_input - noise
                target = model_input

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.training.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.training.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.logging_config.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.training.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.training.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.logging_config.output_dir,
                                                                       removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.logging_config.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if args.val_data.validation_image is not None and global_step % args.val_data.validation_iters == 0:
                    # create pipeline
                    pipeline = StableDiffusion3Pipeline.from_pretrained(
                        args.model.pretrained_model_name_or_path,
                        transformer=accelerator.unwrap_model(transformer),
                        vae=vae,
                        vq_model_embedder=vq_model_embedder,
                        revision=args.model.revision,
                        variant=args.model.variant,
                        torch_dtype=torch.float32,
                    )
                    pipeline.set_prompt_embeds(empty_prompt_embeds, empty_pooled_prompt_embeds)
                    pipeline_args = {
                        "image": args.val_data.validation_image,  # Default prompt for validation
                    }
                    images = log_validation(
                        pipeline=pipeline,
                        args=args,
                        accelerator=accelerator,
                        pipeline_args=pipeline_args,
                        epoch=epoch,
                    )
                    free_memory()

                    images = None
                    del pipeline

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            logs.update(
                {"weight_of_input": transformer.module.pos_embed.proj.weight.data.detach()[:, 32:].norm().item()})
            logs.update(
                {"model_input2_mean": model_input2.mean().item(), "model_input2_std": model_input2.std().item()})
            logs.update({"sigmas": sigmas.mean().item()})
            if grad_norm is not None:
                logs.update({"grad_norm": grad_norm.item()})
            progress_bar.set_postfix(**logs)

            if 'wandb' in args.logging_config.report_to:
                if accelerator.is_main_process:
                    wandb.log(logs, step=global_step)
            accelerator.log(logs, step=global_step)

            if global_step >= args.training.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.val_data.validation_image is not None and global_step % args.val_data.validation_iters == 0:
                # create pipeline
                pipeline = StableDiffusion3Pipeline.from_pretrained(
                    args.model.pretrained_model_name_or_path,
                    transformer=accelerator.unwrap_model(transformer),
                    vae=vae,
                    vq_model_embedder=vq_model_embedder,
                    revision=args.model.revision,
                    variant=args.model.variant,
                    torch_dtype=weight_dtype,
                )
                pipeline.set_prompt_embeds(empty_prompt_embeds, empty_pooled_prompt_embeds)

                pipeline_args = {
                    "image": args.val_data.validation_image,  # Default prompt for validation
                }
                images = log_validation(
                    pipeline=pipeline,
                    args=args,
                    accelerator=accelerator,
                    pipeline_args=pipeline_args,
                    epoch=epoch,
                )
                torch.cuda.empty_cache()
                gc.collect()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)

        pipeline = StableDiffusion3Pipeline.from_pretrained(
            args.model.pretrained_model_name_or_path,
            transformer=transformer,
        )
        pipeline.set_prompt_embeds(empty_prompt_embeds, empty_pooled_prompt_embeds)

        # save the pipeline
        pipeline.save_pretrained(args.logging_config.output_dir)

        # Final inference
        # Load previous pipeline
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            args.logging_config.output_dir,
            revision=args.model.revision,
            variant=args.model.variant,
            torch_dtype=weight_dtype,
        )
        pipeline.set_prompt_embeds(empty_prompt_embeds, empty_pooled_prompt_embeds)

        # run inference
        images = []
        if args.val_data.validation_image:
            pipeline_args = {
                "image": args.val_data.validation_image,  # Default prompt for validation
            }

            images = log_validation(
                pipeline=pipeline,
                args=args,
                accelerator=accelerator,
                pipeline_args=pipeline_args,
                epoch=epoch,
                is_final_validation=True,
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.model.pretrained_model_name_or_path,
                instance_prompt=args.instance_prompt,
                validation_prompt=args.val_data.validation_image,
                repo_folder=args.logging_config.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.logging_config.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
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
import logging
import math
import os
from pathlib import Path
import shutil
from einops import rearrange

import random
import numpy as np
import torch
import torch.utils.checkpoint
import torch.distributed as dist

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, Gemma2Model

import diffusers
from diffusers import (
    AutoencoderDC,
    FlowMatchEulerDiscreteScheduler,
    SanaTransformer2DModel,
)

from diffusers.models.transformers.sana_transformer import GLUMBConv

from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

# Add missing imports
from dataset.build import make_transform, build_dataset
from dataset.multi_ratio_dataset import ResolutionBatchSampler
from utils.registry_utils import read_config
from tokenizer.sana_decoder_pipe import SanaPipeline, DualViTok2ImageEmbedder
from utils.dist_utils import get_rank

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.32.0")

logger = get_logger(__name__)


def transformer_expand_input_channels(transformer, add_in_channels):
    """
    Expand transformer input channels to accommodate additional VQ-GAN features.
    Similar to unet_expand_conv_in but for SANA transformer.
    """
    # For SANA transformer, we need to modify the input projection layer
    # This is typically the first linear layer that processes the input latents
    orig_proj = transformer.patch_embed.proj
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
        new_proj.weight[:, :orig_proj.in_channels] = orig_proj.weight
        # Initialize new channels with small random values
        torch.nn.init.kaiming_normal_(new_proj.weight[:, orig_proj.in_channels:], mode='fan_out')
        if new_proj.bias is not None and orig_proj.bias is not None:
            new_proj.bias.copy_(orig_proj.bias)

    transformer.patch_embed.proj = new_proj

    return transformer


@torch.no_grad
def compute_vae_encodings(batch, vae, dualvitok, diffusion_upsample_rate=1):
    """
    Compute VAE encodings and DualViTok features for VQ-GAN decoder training.
    Adapted from SDXL decoder training script.
    """
    images = batch.pop("pixel_values")
    if images.ndim == 5:
        images = images.squeeze(0)
    pixel_values = torch.stack(list(images))
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)

    # Encode with VAE
    model_input = vae.encode(pixel_values).latent
    if hasattr(vae.config, 'scaling_factor'):
        model_input = model_input * vae.config.scaling_factor

    b, c, h, w = pixel_values.shape
    if diffusion_upsample_rate > 1:
        h2, w2 = h // diffusion_upsample_rate, w // diffusion_upsample_rate
        pixel_values = torch.nn.functional.interpolate(pixel_values, size=(h2, w2), mode='bicubic')

    # Encode with DualViTok
    model_input2 = dualvitok(pixel_values)

    # Ensure spatial dimensions match
    if model_input2.shape[-2:] != model_input.shape[-2:]:
        model_input2 = torch.nn.functional.interpolate(model_input2, size=model_input.shape[-2:])

    batch.update({"model_input": model_input.detach(), "model_input2": model_input2.detach()})
    return batch


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
# Sana DreamBooth LoRA - {repo_id}

<Gallery />

## Model description

These are {repo_id} DreamBooth LoRA weights for {base_model}.

The weights were trained using [DreamBooth](https://dreambooth.github.io/) with the [Sana diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sana.md).


## Trigger words

You should use `{instance_prompt}` to trigger the image generation.

## Download model

[Download the *.safetensors LoRA]({repo_id}/tree/main) in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
TODO
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## License

TODO
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "lora",
        "sana",
        "sana-diffusers",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
        pipeline,
        args,
        accelerator,
        pipeline_args,
        epoch,
        is_final_validation=False,
):
    logger.info("Running validation... Recording image reconstruction effect with VQ-GAN decoder.")

    # Load validation image for reconstruction
    if hasattr(args.val_data, 'validation_image') and args.val_data.validation_image is not None:
        validation_image_path = args.val_data.validation_image
        try:
            original_image = Image.open(validation_image_path).convert('RGB')
            logger.info(f"Loaded validation image from: {validation_image_path}")
        except Exception as e:
            logger.warning(f"Failed to load validation image: {e}")
            return []
    else:
        logger.warning("No validation image specified, skipping reconstruction validation.")
        return []

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # Resize original image to target resolution
    original_image_resized = original_image.resize((args.val_data.resolution // 2, args.val_data.resolution // 2),
                                                   Image.Resampling.LANCZOS)
    target_size = (args.val_data.resolution, args.val_data.resolution)

    # Convert to tensor for processing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    with torch.no_grad():
        # Prepare input for reconstruction using the VQ-GAN decoder pipeline
        # For VQ-GAN decoder, we use the diffusion process for reconstruction
        # This simulates the reconstruction process using the trained decoder

        # Use the pipeline's generation method for reconstruction
        # Note: This assumes the pipeline has been modified to support image-to-image reconstruction
        reconstructed_images = pipeline(
            image=original_image_resized,
            num_inference_steps=25,
            guidance_scale=3.0,
            height=args.val_data.resolution,
            width=args.val_data.resolution,
            generator=torch.Generator(device=accelerator.device).manual_seed(42)
        ).images

        if reconstructed_images:
            reconstructed_image = reconstructed_images[0]
        else:
            # Fallback: just return the original image
            reconstructed_image = original_image_resized

    # Create comparison images
    images_to_log = [original_image_resized.resize(target_size), reconstructed_image]

    # Log to trackers
    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            # Stack original and reconstructed images
            np_images = np.stack([np.asarray(img) for img in images_to_log])
            tracker.writer.add_images(f"{phase_name}/reconstruction", np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    f"{phase_name}/reconstruction": [
                        wandb.Image(images_to_log[0], caption="Original"),
                        wandb.Image(images_to_log[1], caption="Reconstructed")
                    ]
                }
            )

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images_to_log


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

    if torch.backends.mps.is_available() and args.training.mixed_precision == "bf16":
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

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
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

    # Load the tokenizer for empty text embeddings
    tokenizer = AutoTokenizer.from_pretrained(
        args.model.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.model.revision,
    )

    # Initialize text encoder for empty text embeddings
    text_encoder = Gemma2Model.from_pretrained(
        args.model.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.model.revision,
        variant=args.model.variant
    )
    text_encoding_pipeline = SanaPipeline.from_pretrained(
        args.model.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    empty_prompt_embeds, empty_prompt_attention_mask = text_encoding_pipeline.empty_prompt_embeds, text_encoding_pipeline.empty_prompt_attention_mask
    print("empty_prompt_embeds", empty_prompt_embeds.shape, empty_prompt_attention_mask.shape)

    # Clean up text encoder memory
    del text_encoder, text_encoding_pipeline, tokenizer
    free_memory()

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model.pretrained_model_name_or_path, subfolder="scheduler", revision=args.model.revision
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    # text_encoder = Gemma2Model.from_pretrained(
    #     args.model.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.model.revision, variant=args.model.variant
    # )
    vae = AutoencoderDC.from_pretrained(
        args.model.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.model.revision,
        variant=args.model.variant,
    )
    transformer = SanaTransformer2DModel.from_pretrained(
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

    vq_embed_projector = None
    if args.model.use_vq_embed_projector:
        vq_embed_projector = GLUMBConv(
            in_channels=additional_channels,
            out_channels=empty_prompt_embeds.shape[-1],
            expand_ratio=4,
            norm_type='rms_norm',
            residual_connection=False,
        ).to(weight_dtype)

    # Update transformer config
    if hasattr(transformer.config, 'in_channels'):
        transformer.register_to_config(in_channels=transformer.config.in_channels + additional_channels)

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)  # Will be set to True for LoRA layers later
    vae.requires_grad_(False)
    vq_model_embedder.requires_grad_(False)
    if vq_embed_projector is not None:
        vq_embed_projector.requires_grad_(True)

        # VAE should always be kept in fp32 for SANA (?)
    vae.to(dtype=torch.float32)
    vq_model_embedder.to(accelerator.device, dtype=torch.float32)
    transformer.to(accelerator.device, dtype=weight_dtype)

    if args.training.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

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
    transformer.patch_embed.requires_grad_(True)

    if dist.get_rank() == 0:
        for key, value in transformer.named_parameters():
            print(key, value.dtype, value.requires_grad)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        print('save checkpoint to', output_dir)
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            vq_embed_projector_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    patch_embed_to_save = unwrap_model(transformer).patch_embed.state_dict()
                elif isinstance(model, type(unwrap_model(vq_embed_projector))):
                    vq_embed_projector_to_save = unwrap_model(vq_embed_projector).state_dict()
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            SanaPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )
            torch.save(patch_embed_to_save, os.path.join(output_dir, "sana_patch_embed.bin"))
            if vq_embed_projector_to_save:
                torch.save(vq_embed_projector_to_save, os.path.join(output_dir, "sana_vq_embed_projector.bin"))

    def load_model_hook(models, input_dir):
        transformer_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = SanaPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

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

    # Make sure the trainable params are in float32.
    if args.training.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.training.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]
    if vq_embed_projector is not None:
        vq_embed_projector_parameters = list(filter(lambda p: p.requires_grad, vq_embed_projector.parameters()))
        vq_embed_projector_parameters_with_lr = {"params": vq_embed_projector_parameters,
                                                 "lr": args.training.projector_learning_rate}
        params_to_optimize += [vq_embed_projector_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.type.lower() == "prodigy" or args.optimizer.type.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer.type}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer.type = "adamw"

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
            betas=(args.optimizer.adam_beta1, args.optimizer.adam_beta2),
            beta3=args.optimizer.prodigy_beta3,
            weight_decay=args.optimizer.adam_weight_decay,
            eps=args.optimizer.adam_epsilon,
            decouple=args.optimizer.prodigy_decouple,
            use_bias_correction=args.optimizer.prodigy_use_bias_correction,
            safeguard_warmup=args.optimizer.prodigy_safeguard_warmup,
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

    # Configure dataloader based on batch sampler
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

    vae_config_scaling_factor = vae.config.scaling_factor
    if args.training.cache_latents:
        latents_cache = []
        vae = vae.to(accelerator.device)
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(
                    accelerator.device, non_blocking=True, dtype=vae.dtype
                )
                latents_cache.append(vae.encode(batch["pixel_values"]).latent)

        if args.val_data.validation_image is None:
            del vae
            free_memory()

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

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    if vq_embed_projector is not None:
        vq_embed_projector = accelerator.prepare(
            vq_embed_projector
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
        tracker_name = "dualvitok-sana-decoder-lora"
        accelerator.init_trackers(tracker_name)

    # Train!
    total_batch_size = args.data.train_batch_size * accelerator.num_processes * args.training.gradient_accumulation_steps

    logger.info("***** Running VQ-GAN Decoder Training with SANA Transformer *****")
    logger.info("Training objective: Image reconstruction using VAE + DualViTok features")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.training.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.data.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.training.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.training.max_train_steps}")
    logger.info(f"  DualViTok channels: {vq_model_embedder.pixel_channel + vq_model_embedder.semantic_channel}")
    global_step = 0
    first_epoch = 0

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
                # Use pre-computed empty text embeddings
                prompt_embeds = empty_prompt_embeds.repeat(batch["pixel_values"].shape[0], 1, 1)
                prompt_attention_mask = empty_prompt_attention_mask.repeat(batch["pixel_values"].shape[0], 1)

                # Compute VAE encodings and DualViTok features
                vae = vae.to(accelerator.device)
                batch = compute_vae_encodings(batch, vae, vq_model_embedder,
                                              diffusion_upsample_rate=args.model.diffusion_upsample_rate)
                if args.training.offload:
                    vae = vae.to("cpu")

                model_input = batch["model_input"].to(accelerator.device, dtype=weight_dtype)
                model_input2 = batch["model_input2"].to(accelerator.device, dtype=weight_dtype)

                # Zero out VQ-GAN features if specified (for ablation)
                if hasattr(args.training, 'proportion_empty_prompts') and args.training.proportion_empty_prompts:
                    if random.random() < args.training.proportion_empty_prompts:
                        model_input2 = torch.zeros_like(model_input2)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.training.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.training.logit_mean,
                    logit_std=args.training.logit_std,
                    mode_scale=args.training.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                # Concatenate noised VAE features with DualViTok features
                # print(noisy_model_input.mean(), noisy_model_input.std(), model_input2.mean(), model_input2.std())
                transformer_input = torch.cat([noisy_model_input, model_input2], dim=1)

                if vq_embed_projector is not None:
                    model_input_condition = rearrange(vq_embed_projector(model_input2), 'b c h w -> b (h w) c')
                    encoder_hidden_states = model_input_condition
                    encoder_attention_mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.long)
                else:
                    encoder_hidden_states = prompt_embeds
                    encoder_attention_mask = prompt_attention_mask

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=transformer_input,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timesteps,
                    return_dict=False,
                )[0]

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.training.weighting_scheme,
                                                           sigmas=sigmas)

                # Flow matching loss for VQ-GAN decoder training
                # Target is the difference between noise and original latent
                target = noise - model_input

                # Compute loss
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)  # bugs here
                # accelerator.backward(loss, retain_graph=True)  # bugs here
                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.training.max_grad_norm)

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

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.training.max_train_steps:
                break

            if accelerator.is_main_process:

                if args.val_data.validation_image is not None and global_step % args.val_data.validation_iters == 0:
                    # create pipeline
                    pipeline = SanaPipeline.from_pretrained(
                        args.model.pretrained_model_name_or_path,
                        transformer=accelerator.unwrap_model(transformer),
                        vae=vae,
                        vq_model_embedder=vq_model_embedder,
                        revision=args.model.revision,
                        variant=args.model.variant,
                        torch_dtype=torch.float32,
                    )
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

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        if args.training.upcast_before_saving:
            transformer.to(torch.float32)
        else:
            transformer = transformer.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)

        SanaPipeline.save_lora_weights(
            save_directory=args.logging_config.output_dir,
            transformer_lora_layers=transformer_lora_layers,
        )

        # Final inference
        # Load previous pipeline
        pipeline = SanaPipeline.from_pretrained(
            args.model.pretrained_model_name_or_path,
            revision=args.model.revision,
            variant=args.model.variant,
            torch_dtype=torch.float32,
        )
        pipeline.transformer = pipeline.transformer.to(torch.float16)
        # load attention processors
        pipeline.load_lora_weights(args.logging_config.output_dir)

        # run inference
        images = []
        if hasattr(args.val_data, 'validation_image') and args.val_data.validation_image is not None:
            pipeline_args = {
                "prompt": "",  # Empty prompt for reconstruction task
                "complex_human_instruction": None,
            }
            images = log_validation(
                pipeline=pipeline,
                args=args,
                accelerator=accelerator,
                pipeline_args=pipeline_args,
                epoch=args.training.num_train_epochs,
                is_final_validation=True,
            )

        if args.logging_config.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.model.pretrained_model_name_or_path,
                instance_prompt="VQ-GAN reconstruction",
                validation_prompt="Image reconstruction using VQ-GAN decoder",
                repo_folder=args.logging_config.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.logging_config.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        images = None
        del pipeline

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)

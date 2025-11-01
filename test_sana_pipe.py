import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, SanaTransformer2DModel
from diffusers.models.transformers.sana_transformer import GLUMBConv

from transformers import BitsAndBytesConfig as BitsAndBytesConfig, AutoModel
from PIL import Image
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from peft import LoraConfig, set_peft_model_state_dict, PeftModel

import numpy as np
import os

from dataset.build import make_transform, build_dataset
from utils.registry_utils import read_config
from tokenizer.sana_decoder_pipe import SanaPipeline, DualViTok2ImageEmbedder
from safetensors.torch import load_file
import argparse


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
    ).to(device=orig_proj.weight.data.device, dtype=orig_proj.weight.data.dtype)

    # Copy original weights
    with torch.no_grad():
        new_proj.weight[:, :orig_proj.in_channels] = orig_proj.weight
        # Initialize new channels with small random values
        torch.nn.init.kaiming_normal_(new_proj.weight[:, orig_proj.in_channels:], mode='fan_out')
        if new_proj.bias is not None and orig_proj.bias is not None:
            new_proj.bias.copy_(orig_proj.bias)

    transformer.patch_embed.proj = new_proj

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


if __name__ == "__main__":
    args = parse_args()

    dualvitok_config = '../configs/example/dualvitok/dualvitok_anyres_max512.py'
    vq_config = read_config(dualvitok_config)
    vq_model_embedder = DualViTok2ImageEmbedder(
        vq_config, vq_ckpt=args.model.dualvitok_ckpt)
    validation_image = "/mnt/hdfs/user/huangrunhui/data/huggingface_datasets/BLIP3o-60k/dalle3/1.jpg"
    original_image = Image.open(validation_image)
    resolution = 1024
    original_image_resized = original_image.resize((resolution // 2, resolution // 2))
    target_size = (resolution, resolution)
    additional_channels = vq_model_embedder.pixel_channel + vq_model_embedder.semantic_channel

    # vq_model_embedder = vq_model_embedder.to(torch.float32)
    # with torch.autocast('cuda', dtype=torch.float32):
    #     quant_semantic, quant_detail = vq_model_embedder.vq_encode_code([original_image_resized])
    #     reconc_image = vq_model_embedder.vq_decode_code(quant_semantic, quant_detail)
    #     print(reconc_image.max(), reconc_image.min())
    #     # reconc_image = reconc_image / 1.25
    #     Image.fromarray((255 * reconc_image).detach().float().cpu()[0].permute(1, 2, 0).numpy().astype("uint8")).save('vq_reconstructed.png')

    text_encoder_8bit = AutoModel.from_pretrained(
        "/mnt/hdfs/user/huangrunhui/huggingface_models/SANA1.5_1.6B_1024px_diffusers",
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
    )

    transformer_8bit = SanaTransformer2DModel.from_pretrained(
        "/mnt/hdfs/user/huangrunhui/huggingface_models/SANA1.5_1.6B_1024px_diffusers",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    vq_embed_projector = None
    if args.model.use_vq_embed_projector:
        vq_embed_projector = GLUMBConv(
            in_channels=additional_channels,
            out_channels=2304,  # fix embeds. should read from config?
            expand_ratio=4,
            norm_type='rms_norm',
            residual_connection=False,
        ).cuda()

    pipeline = SanaPipeline.from_pretrained(
        "/mnt/hdfs/user/huangrunhui/huggingface_models/SANA1.5_1.6B_1024px_diffusers",
        text_encoder=text_encoder_8bit,
        transformer=transformer_8bit,
        vq_config=vq_config,
        vq_model_embedder=vq_model_embedder,
        vq_embed_projector=vq_embed_projector,
        torch_dtype=torch.bfloat16,
    )

    pipeline.text_encoder = text_encoder_8bit

    # Update transformer config
    if hasattr(pipeline.transformer.config, 'in_channels'):
        pipeline.transformer.register_to_config(
            in_channels=pipeline.transformer.config.in_channels + additional_channels)
    pipeline.transformer = transformer_expand_input_channels(pipeline.transformer, additional_channels)

    # load lora
    checkpoint_dir = os.path.join(args.logging_config.output_dir, 'checkpoint-650')
    lora_state_dict = SanaPipeline.lora_state_dict(checkpoint_dir)

    target_modules = ["to_k", "to_q", "to_v", "to_out.0",
                      "conv_point",
                      #   "conv_depth",
                      "conv_inverted"
                      ]

    transformer_lora_config = LoraConfig(
        r=args.lora.rank,
        lora_alpha=args.lora.rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    pipeline.transformer.add_adapter(transformer_lora_config)

    transformer_state_dict = {
        f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
    }
    transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
    incompatible_keys = set_peft_model_state_dict(pipeline.transformer, transformer_state_dict, adapter_name="default")
    # pipeline.transformer.merge_and_unload()
    print(incompatible_keys)

    # load patch embeds
    patch_embed_path = os.path.join(checkpoint_dir, 'sana_patch_embed.safetensors')
    if os.path.exists(patch_embed_path):
        patch_embed_state_dict = torch.load(patch_embed_path, 'cpu')
        pipeline.transformer.patch_embed.load_state_dict(patch_embed_state_dict)

    embed_projector_path = os.path.join(checkpoint_dir, 'sana_vq_embed_projector.safetensors')
    if os.path.exists(embed_projector_path):
        embed_projector_state_dict = torch.load(embed_projector_path, 'cpu')
        pipeline.transformer.vq_embed_projector.load_state_dict(embed_projector_state_dict)

    pipeline.to('cuda')

    reconstructed_images = pipeline(
        image=original_image_resized,
        num_inference_steps=50,
        guidance_scale=3.0,
        height=resolution,
        width=resolution,
        generator=torch.Generator(device='cuda').manual_seed(42)
    ).images
    if reconstructed_images:
        reconstructed_image = reconstructed_images[0]
    else:
        # Fallback: just return the original image
        reconstructed_image = original_image_resized
    original_image_resized.save('sana_original.png')
    reconstructed_image.save('sana_reconstructed.png')

    # pipeline.save_pretrained('/opt/tiger/dualvitok-sana-decoder')


from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)




def import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


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


def load_model_hook(args, model, input_dir):
    if args.lora.use_lora:
        from peft import LoraConfig, set_peft_model_state_dict
        lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(model, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
        if os.path.exists(os.path.join(input_dir, "sd3_pos_embed.bin")):
            model.pos_embed.load_state_dict(torch.load(os.path.join(input_dir, "sd3_pos_embed.bin"), 'cpu'))

        shift_table_path = os.path.join(input_dir, 'sd3_scale_shift_table.bin')
        if os.path.exists(shift_table_path):
            model.load_state_dict(torch.load(shift_table_path, 'cpu'), strict=False)
    else:
        load_model = SD3Transformer2DModel.from_pretrained(input_dir, subfolder="transformer")
        model.register_to_config(**load_model.config)
        model.load_state_dict(load_model.state_dict())
    return model


def load_pretrained_sd3_decoder_from_config(args, decoder_ckpt_path, vq_config, vq_model):
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


    def load_text_encoders(class_one, class_two, class_three):
        text_encoder_one = class_one.from_pretrained(
            args.model.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.model.revision, variant=args.model.variant
        )
        text_encoder_two = class_two.from_pretrained(
            args.model.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.model.revision, variant=args.model.variant
        )
        text_encoder_three = class_three.from_pretrained(
            args.model.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.model.revision, variant=args.model.variant
        )
        return text_encoder_one, text_encoder_two, text_encoder_three

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
        args.model.pretrained_model_name_or_path, subfolder="transformer", revision=args.model.revision, variant=args.model.variant
    )

    vq_model_embedder = DualViTok2ImageEmbedder(vq_config, vq_model=vq_model)

    additional_channels = vq_model_embedder.pixel_channel + vq_model_embedder.semantic_channel
    print(f'Expanding transformer input channels by {additional_channels} for DualViTok features')
    transformer = transformer_expand_input_channels(transformer, additional_channels)

    # Update transformer config
    if hasattr(transformer.config, 'in_channels'):
        transformer.register_to_config(in_channels=transformer.config.in_channels + additional_channels)

    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]
    with torch.no_grad():
        empty_prompt_embeds, empty_pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, '')

    if args.lora.use_lora:
        if args.lora.lora_layers is not None:
            target_modules = [layer.strip() for layer in args.lora.lora_layers.split(",")]
        else:
            target_modules = [
                "to_k", "to_q", "to_v", "to_out.0",
                "ff_context.net.0.proj",
                "ff.net.0.proj",
            ]

        from peft import LoraConfig

        # now we will add new LoRA weights the transformer layers
        transformer_lora_config = LoraConfig(
            r=args.lora.rank,
            lora_alpha=args.lora.rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        transformer.add_adapter(transformer_lora_config)

    # load decoder pretrain weight
    transformer = load_model_hook(args, transformer, decoder_ckpt_path)
    weight_dtype = torch.bfloat16

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        args.model.pretrained_model_name_or_path,
        transformer=transformer,
        vae=vae,
        vq_model_embedder=vq_model_embedder,
        revision=args.model.revision,
        variant=args.model.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.set_prompt_embeds(empty_prompt_embeds, empty_pooled_prompt_embeds)
    pipeline.to('cuda')

    return pipeline


    if args.use_diffusion_decoder:
        if args.diffusion_decoder_type == 'sd3':
            diffusion_decoder_pipe = load_pretrained_sd3_decoder_from_config(
                args.diffusion_decoder_config, args.diffusion_decoder_ckpt,
                vq_config=config,
                vq_model=vq_model)
        elif args.diffusion_decoder_type == 'sana':
            raise NotImplementedError("SANA decoder is not implemented yet.")
        elif args.diffusion_decoder_type == 'sdxl':
            from tokenizer.sdxl_decoder_pipe import StableDiffusionXLDecoderPipeline
            diffusion_decoder_pipe = StableDiffusionXLDecoderPipeline.from_pretrained(
                args.diffusion_decoder_ckpt,
                torch_dtype=args.torch_dtype,
                add_watermarker=False,
                vq_config=config,
                vq_model=vq_model,
            )
            diffusion_decoder_pipe.enable_model_cpu_offload()


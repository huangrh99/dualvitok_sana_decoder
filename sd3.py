# config.py
"""
Reorganized mmcv-style configuration file for the training script.
Each section corresponds to a logical domain, making it easier to manage and extend.
"""

import os

# ---------------------------------------------------
# Model Configurations
# ---------------------------------------------------
model = dict(
    # Model parameters
    # pretrained_model_name_or_path='/cache/data/huggingface_models/stable-diffusion-xl-base-1.0',  # Required: override externally
    # pretrained_vae_model_name_or_path='/cache/data/huggingface_models/sdxl-vae-fp16-fix/',
    # dualvitok_config='../configs/example/dualvitok/dualvitok_stage3_anyres_max512.py',
    # dualvitok_ckpt='/cache/exps/dualvitok/dualvitok_stage3_31w.pt',

    pretrained_model_name_or_path='/mnt/hdfs/user/huangrunhui/huggingface_models/stable-diffusion-3-medium-diffusers',
    dualvitok_config='../configs/example/dualvitok/dualvitok_anyres_max512.py',
    dualvitok_ckpt='/mnt/hdfs/user/huangrunhui/huggingface_models/dualvitok/pytorch_model.bin',

    use_vq_embed_projector=False,

    revision=None,
    variant=None,
    train_part='all',  # Specify which part of the model to train
    prediction_type=None,  # Options: 'epsilon', 'v_prediction', or None (default from scheduler)
    # Tokenizer confi；guration for discrete vision (expand as needed)
    condition='concat',
    diffusion_upsample_rate=2,
)

resolution_list = [
    # (256, 256), (256, 384), (384, 256),
    (1024, 1024), (768, 1024), (1024, 768),
    (512, 2048), (2048, 512), (640, 1920),
    (1920, 640), (768, 1536),
    (1536, 768), (768, 1152), (1152, 768)
]

# ---------------------------------------------------
# Dataset Configurations
# ---------------------------------------------------
data = dict(
    resolution=1024,  # Resize all input images to this resolution
    train_batch_size=8,

    dataloader_num_workers=4,  # Number of subprocesses for data loading; 0 means loading in the main process

    # huggingface dataset
    # dataset_name='/cache/data/huggingface_datasets/lambdalabs/naruto-blip-captions',         # Name of the dataset (e.g., from Hugging Face); can be None if not used
    # dataset_config_name=None,  # Dataset configuration name; set to None if there's only one config
    # train_data_dir=None,       # Path to the training data directory (required if dataset_name is None)
    # image_column='image',      # Column name containing images
    # caption_column='text',     # Column name containing captions
    center_crop=True,  # Whether to center crop images
    random_flip=True,  # Whether to randomly flip images horizontally

    ## args for multi-resolution dataset.
    dataset_name='multires',
    dataset_args=dict(
        resolution=resolution_list,
        augment=dict(type='multi_resolution_random_crop_flip'),

        dataset=[
            dict(dataset='multires',
                 data_path='/mnt/hdfs/user/huangrunhui/data/huggingface_datasets/BLIP3o-60k',
                 json_file='/mnt/hdfs/user/huangrunhui/data/huggingface_datasets/BLIP3o-60k/blip3o_60k_meta.json',
                 #  json_file='/mnt/hdfs/user/huangrunhui/data/huggingface_datasets/BLIP3o-60k/blip3o_1_meta.json',
                 #  json_file='json_files/imagenet_train.json',
                 shard_data=False, global_sharding=True,
                 base_sizes=resolution_list,
                 ),

            # imagenet
            # dict(dataset='multires',
            #      data_path='/cache/data/ILSVRC/Data/CLS-LOC/train/',
            #      json_file='json_files/imagenet_train.json',
            #      shard_data=False, global_sharding=True,
            #      base_sizes=resolution_list,
            # ),

            # # emova image understanding. Pretrain data.
            # dict(dataset='multires',
            #      data_path="s3://bucket-pangu-green-guiyang/huangrunhui/data/emova_pretrain/",
            #      json_file='/cache/data/vae_json_files/emova_pretrain.json',
            #      shard_data=False, global_sharding=True),

        ],

    )

)

# ---------------------------------------------------
# Validation Configurations
# ---------------------------------------------------
val_data = dict(
    resolution=1024,

    # Prompt used during validation for image generation

    # validation_image="../configs/example/diffusion_decoder/example.png",
    validation_image="/mnt/hdfs/user/huangrunhui/data/huggingface_datasets/BLIP3o-60k/dalle3/1.jpg",
    # Default prompt for validation

    # validation_image="../configs/example/diffusion_decoder/example.png",
    num_validation_images=1,  # Number of images to generate for validation
    validation_iters=100,  # Frequency of validation (every X epochs)
)

# ---------------------------------------------------
# Training Process Configurations
# ---------------------------------------------------
training = dict(
    max_train_samples=None,  # Truncate training samples for debugging or faster training
    proportion_empty_prompts=0.1,  # Proportion of image prompts replaced with empty strings (range [0, 1])

    seed=423,
    # deepspeed_config=None,
    deepspeed_config='scripts/zero2.json',

    # "The timestep bias strategy, which may help direct the model toward learning low or high frequency details."
    # " Choices: ['earlier', 'later', 'range', 'none']."
    timestep_bias_strategy='none',  # Options: 'earlier', 'later', 'range', 'none'.
    timestep_bias_multiplier=1.0,
    timestep_bias_begin=0,
    timestep_bias_end=1000,
    timestep_bias_portion=0.25,

    num_train_epochs=10000,
    max_train_steps=None,  # If provided, overrides num_train_epochs
    checkpointing_steps=500,  # Save a checkpoint every X steps
    checkpoints_total_limit=10,  # Maximum number of checkpoints to store
    resume_from_checkpoint='latest',  # Checkpoint path to resume training, or "latest" for the most recent one
    gradient_accumulation_steps=1,  # Number of steps to accumulate gradients before updating
    gradient_checkpointing=False,  # Enable gradient checkpointing to save memory
    mixed_precision='bf16',  # Mixed precision mode: 'no', 'fp16', or 'bf16'
    enable_npu_flash_attention=False,  # Enable NPU flash attention if available
    enable_xformers_memory_efficient_attention=True,  # Enable xformers memory efficient attention if available

    noise_offset=0,  # Scale of noise offset

    lr_scheduler='constant',
    # Options: 'linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'
    lr_warmup_steps=10,  # Number of steps for learning rate warmup
    lr_num_cycles=1,  # Number of hard resets of the lr in cosine_with_restarts scheduler
    lr_power=1.0,  # Power factor of the polynomial scheduler

    snr_gamma=None,  # SNR weighting gamma (e.g., recommended value: 5.0)
    use_ema=False,  # Whether to use EMA for the model
    allow_tf32=True,  # Allow TF32 to speed up training on Ampere GPUs
    use_8bit_adam=False,  # Use 8-bit Adam optimizer from bitsandbytes

    upcast_before_saving=False,
    offload=False,

    max_grad_norm=1.0,  # Maximum gradient norm for clipping
    learning_rate=1e-4,
    scale_lr=False,  # Scale learning rate by the number of GPUs, accumulation steps, and batch size

    # Weighting schemes for timestep sampling
    weighting_scheme='none',  # Options: 'sigma_sqrt', 'logit_normal', 'mode', 'cosmap', 'none'
    logit_mean=0.0,  # Mean for logit_normal weighting scheme
    logit_std=1.0,  # Standard deviation for logit_normal weighting scheme
    mode_scale=1.29,  # Scale for mode weighting scheme

    # Cache latents for faster training
    cache_latents=False,  # Cache the VAE latents

    tune_scale_shift=False,
)

# ---------------------------------------------------
# LoRA Configurations
# ---------------------------------------------------
lora = dict(
    use_lora=True,
    rank=128,  # The dimension of the LoRA update matrices
    lora_layers=None,  # Comma-separated list of transformer modules to apply LoRA on
    lora_pretrained_path=False,
    merge_lora=False,
)

# ---------------------------------------------------
# Optimizer and Scheduler Configurations
# ---------------------------------------------------
optimizer = dict(
    type='AdamW',  # Options: 'AdamW', 'prodigy'
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_weight_decay=0.,
    adam_epsilon=1e-08,

    # Prodigy optimizer specific parameters
    prodigy_beta3=None,  # Coefficients for computing Prodigy stepsize
    prodigy_decouple=True,  # Use AdamW style decoupled weight decay
    prodigy_use_bias_correction=True,  # Turn on Adam's bias correction
    prodigy_safeguard_warmup=True,  # Remove lr from denominator during warmup
)

# ---------------------------------------------------
# Logging, Hub, and Mixed Precision Configurations
# ---------------------------------------------------
logging_config = dict(
    push_to_hub=False,
    hub_token=None,
    hub_model_id=None,  # Repository name for syncing the model to the Hub
    output_dir='/mnt/hdfs/user/huangrunhui/exps/illume_sana/illume_plus-sd3-decoder-v1-1r1e-4-60k',
    logging_dir='logs',  # Directory for storing logs (e.g., TensorBoard)
    report_to=['tensorboard', 'wandb'],  # Reporting platform: 'tensorboard', 'wandb', 'comet_ml', or 'all'
    run_name='sd3-decoder-v1-lr1e-4'
)

# ---------------------------------------------------
# Hardware Acceleration Configurations
# ---------------------------------------------------

# ---------------------------------------------------
# Basic Sanity Checks (Optional)
# ---------------------------------------------------
if data['dataset_name'] is None and data['train_data_dir'] is None:
    raise ValueError("Either 'dataset_name' or 'train_data_dir' must be specified.")
if not (0 <= training['proportion_empty_prompts'] <= 1):
    raise ValueError("proportion_empty_prompts must be in the range [0, 1].")

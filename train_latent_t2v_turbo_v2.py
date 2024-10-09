#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
from copy import deepcopy
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version

from tqdm.auto import tqdm

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available

from data.openvid_data import MP4Dataset

from lvdm.modules.attention import TemporalTransformer
from ode_solver import DDIMSolver
from reward_fn import get_reward_fn
from scheduler.t2v_turbo_scheduler import T2VTurboScheduler
from pipeline.t2v_turbo_vc2_pipeline import T2VTurboVC2Pipeline
from utils.common_utils import (
    append_dims,
    extract_into_tensor,
    get_predicted_noise,
    get_predicted_original_sample,
    guidance_scale_embedding,
    huber_loss,
    log_validation_video,
    scalings_for_boundary_conditions,
    tuple_type,
    load_model_checkpoint,
    update_ema,
)
from utils.utils import instantiate_from_config


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0.dev0")

logger = get_logger(__name__)


@torch.no_grad()
def log_validation(pretrained_t2v, unet, scheduler, model_config, args, accelerator):
    torch.cuda.empty_cache()
    logger.info("Running validation... ")
    pretrained_t2v.model.diffusion_model = unet
    pretrained_t2v.first_stage_model.to(accelerator.device)
    pretrained_t2v.cond_stage_model.to(accelerator.device)
    pipeline = T2VTurboVC2Pipeline(pretrained_t2v, scheduler, model_config)
    pipeline = pipeline.to(accelerator.device)

    log_validation_video(pipeline, args, accelerator, save_fps=args.fps)
    if (
        accelerator.process_index in args.reward_train_processes
        and args.reward_scale > 0
    ) or (
        accelerator.process_index in args.video_rm_train_processes
        and args.video_reward_scale > 0
    ):
        pretrained_t2v.first_stage_model.to(accelerator.device)
    else:
        pretrained_t2v.first_stage_model.to("cpu")

    if not args.train_text_encoder:
        pretrained_t2v.cond_stage_model.to("cpu")

    torch.cuda.empty_cache()
    gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # ----------Model Checkpoint Loading Arguments----------
    parser.add_argument(
        "--pretrained_model_cfg",
        type=str,
        default="configs/inference_t2v_512_v2.0.yaml",
        help="Pretrained Model Config.",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="model_cache/VideoCrafter2_model.ckpt",
        help="Path to the pretrained model.",
    )
    parser.add_argument(
        "--pretrained_unet_dir",
        type=str,
        default=None,
        help="Directory of the pretrained UNet model",
    )
    # ----------Training Arguments----------
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder.",
    )
    parser.add_argument(
        "--unlocked_text_layers",
        type=int,
        default=4,
        help="Number of text layers to unlock.",
    )

    # ----General Training Arguments----
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/t2v-turbo-vc2",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=453645634, help="A seed for reproducible training."
    )
    # ----Logging----
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    # ----Checkpointing----
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=5,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    # ---- Datasets ----
    parser.add_argument(
        "--train_shards_path_or_url",
        type=str,
        default="PATH_TO_DATA_DIR",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--latent_root",
        type=str,
        default="path/to/latent_root",
        help="The root directory of the latent data.",
    )
    # ----Dataloader----
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    # ----Batch Size and Training Steps----
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--reward_frame_bsz",
        type=int,
        default=2,
        help="Batch size (per device) for optimizing the text-image RM.",
    )
    parser.add_argument(
        "--reward_train_bsz",
        type=int,
        default=2,
        help="Batch size (per device) for optimizing the text-image RM.",
    )
    parser.add_argument(
        "--video_rm_frame_bsz",
        type=int,
        default=8,
        help="Batch size (per device) for optimizing the text-image RM.",
    )
    parser.add_argument(
        "--video_rm_train_bsz",
        type=int,
        default=1,
        help="Batch size (per device) for optimizing the text-image RM.",
    )
    parser.add_argument(
        "--vlcd_processes",
        type=tuple_type,
        default=(0, 1, 2, 3, 4, 5),
        help="Process idx that are used to perform consistency distillation.",
    )
    parser.add_argument(
        "--reward_train_processes",
        type=tuple_type,
        default=(0, 1, 2, 3, 4, 5),
        help="Process idx that are used to maximize text-img reward fn.",
    )
    parser.add_argument(
        "--video_rm_train_processes",
        type=tuple_type,
        default=(6, 7),
        help="Process idx that are used to maximize text-video reward fn.",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=16,
        help="Number of frames to sample from a video.",
    )

    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=8000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=1000000,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    # ----Learning Rate----
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    # ----Optimizer (Adam)----
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )

    # ---- Latent Consistency Distillation (LCD) Specific Arguments ----
    parser.add_argument(
        "--w_min",
        type=float,
        default=5.0,
        required=False,
        help=(
            "The minimum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
            " formulation rather than the LCM formulation, which means all guidance scales have 1 added to them as"
            " compared to the original paper."
        ),
    )
    parser.add_argument(
        "--w_max",
        type=float,
        default=15.0,
        required=False,
        help=(
            "The maximum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
            " formulation rather than the LCM formulation, which means all guidance scales have 1 added to them as"
            " compared to the original paper."
        ),
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help=("Eta for solving the DDIM step."),
    )
    parser.add_argument(
        "--no_scale_pred_x0",
        action="store_true",
        default=False,
        help=("Whether to scale the pred_x0 in DDIM step."),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="fps for the video.",
    )
    parser.add_argument(
        "--num_ddim_timesteps",
        type=int,
        default=200,
        help="Num timesteps for DDIM sampling",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="1000 (Num Train timesteps) // 50 (Num timesteps for DDIM sampling)",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="huber",
        choices=["l2", "huber"],
        help="The type of loss to use for the LCD loss.",
    )
    parser.add_argument(
        "--huber_c",
        type=float,
        default=0.001,
        help="The huber loss parameter. Only used if `--loss_type=huber`.",
    )
    parser.add_argument(
        "--unet_time_cond_proj_dim",
        type=int,
        default=256,
        help=(
            "The dimension of the guidance scale embedding in the U-Net, which will be used if the teacher U-Net"
            " does not have `time_cond_proj_dim` set."
        ),
    )
    parser.add_argument(
        "--motion_cond_proj_dim",
        type=int,
        default=256,
        help="The dimension of the motion guidance scale embedding in the U-Net.",
    )
    parser.add_argument(
        "--timestep_scaling_factor",
        type=float,
        default=10.0,
        help=(
            "The multiplicative timestep scaling factor used when calculating the boundary scalings for LCM. The"
            " higher the scaling is, the lower the approximation error, but the default value of 10.0 should typically"
            " suffice."
        ),
    )
    # ----Exponential Moving Average (EMA)----
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.95,
        required=False,
        help="The exponential moving average (EMA) rate or decay factor.",
    )
    # ----Mixed Precision----
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    # ----Training Optimizations----
    parser.add_argument(
        "--use_target_unet",
        action="store_true",
        default=False,
        help="Whether to use a target U-Net for the LCD loss.",
    )
    parser.add_argument(
        "--temporal_lr_scale",
        type=float,
        default=1.0,
        help="The scale of the learning rate for the temporal transformer.",
    )

    # ----Distributed Training----
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    # ----------Validation Arguments----------
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help="Run validation every X steps.",
    )
    # ----------Accelerate Arguments----------
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="t2v-turbo",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--reward_fn_name",
        type=str,
        default="weighted_hpsv2_clip",
        help="Reward function name",
    )
    parser.add_argument(
        "--reward_scale",
        type=float,
        default=0.0,
        help="The scale of the reward loss",
    )
    parser.add_argument(
        "--video_rm_name",
        type=str,
        default="vi_clip2",
        help="Video Reward Model name",
    )
    parser.add_argument(
        "--video_rm_ckpt_dir",
        type=str,
        default="model_cache/InternVideo2-stage2_1b-224p-f4.pt",
        help="Reward function name",
    )
    parser.add_argument(
        "--video_reward_scale",
        type=float,
        default=0.0,
        help="The scale of the viclip reward loss",
    )
    parser.add_argument(
        "--reward_weights",
        type=tuple_type,
        default=(1.0, 1.0),
        help="The scale of the reward loss",
    )
    parser.add_argument(
        "--percentage",
        type=float,
        default=0.5,
        help="Percentage of steps to apply motion guidance",
    )
    parser.add_argument(
        "--motion_gs",
        type=float,
        default=0.05,
        help="The scale of the motion guidance loss",
    )
    parser.add_argument(
        "--use_motion_cond",
        action="store_true",
        default=False,
        help="Whether to use motion guidance scale as input to the U-Net",
    )

    parser.add_argument("--local-rank", type=int, default=1)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.video_rm_name == "vi_clip":
        assert args.video_rm_frame_bsz == 8
    elif args.video_rm_name == "vi_clip2":
        assert args.video_rm_frame_bsz in [4, 8]
    else:
        raise ValueError(f"Unsupported viclip reward function: {args.video_rm_name}")

    if args.reward_train_bsz > args.train_batch_size:
        raise ValueError(
            "Reward training batch size must be less than or equal to the training batch size."
        )
    return args


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        split_batches=True,  # It's important to set this to True when using webdataset to get the right number of steps for lr scheduling. If set to False, the number of steps will be devide by the number of processes assuming batches are multiplied by the number of processes
    )

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
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # 9. Handle mixed precision and device placement
    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    device = accelerator.device

    # 5. Load teacher Model
    config = OmegaConf.load(args.pretrained_model_cfg)
    model_config = config.pop("model", OmegaConf.create())
    pretrained_t2v = instantiate_from_config(model_config)
    pretrained_t2v = load_model_checkpoint(
        pretrained_t2v,
        args.pretrained_model_path,
    )

    vae = pretrained_t2v.first_stage_model
    vae_scale_factor = model_config["params"]["scale_factor"]
    text_encoder = pretrained_t2v.cond_stage_model
    teacher_unet = pretrained_t2v.model.diffusion_model

    # 6. Freeze teacher vae, text_encoder, and teacher_unet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    teacher_unet.requires_grad_(False).eval()

    # 7. Create online student U-Net. This will be updated by the optimizer (e.g. via backpropagation.)
    # Add `time_cond_proj_dim` to the student U-Net if `teacher_unet.config.time_cond_proj_dim` is None
    time_cond_proj_dim = (
        teacher_unet.time_cond_proj_dim
        if teacher_unet.time_cond_proj_dim is not None
        else args.unet_time_cond_proj_dim
    )
    unet_config = model_config["params"]["unet_config"]
    unet_config["params"]["time_cond_proj_dim"] = time_cond_proj_dim
    if args.use_motion_cond:
        unet_config["params"]["motion_cond_proj_dim"] = args.motion_cond_proj_dim
    unet = instantiate_from_config(unet_config)
    # load teacher_unet weights into unet
    if args.pretrained_unet_dir is not None:
        unet.load_state_dict(
            torch.load(args.pretrained_unet_dir, map_location=device, weights_only=True)
        )
    else:
        unet.load_state_dict(teacher_unet.state_dict(), strict=False)
    unet.requires_grad_(True).train()

    if args.use_target_unet:
        unet_config["params"]["use_checkpoint"] = False
        target_unet = instantiate_from_config(unet_config)
        target_unet.load_state_dict(unet.state_dict())
        target_unet.requires_grad_(False).train().to(device)

    del teacher_unet
    torch.cuda.empty_cache()
    if (
        accelerator.process_index in args.reward_train_processes
        and args.reward_scale > 0
    ):
        kwargs = dict(precision=args.mixed_precision)
        if "weighted" in args.reward_fn_name:
            kwargs["weights"] = args.reward_weights
        reward_fn = get_reward_fn(args.reward_fn_name, **kwargs)
    if (
        accelerator.process_index in args.video_rm_train_processes
        and args.video_reward_scale > 0
    ):
        video_rm_fn = get_reward_fn(
            args.video_rm_name,
            precision=args.mixed_precision,
            rm_ckpt_dir=args.video_rm_ckpt_dir,
            n_frames=args.video_rm_frame_bsz,
        )

    # 1. Create the noise scheduler and the desired noise schedule.
    noise_scheduler = T2VTurboScheduler(
        linear_start=model_config["params"]["linear_start"],
        linear_end=model_config["params"]["linear_end"],
    )

    # DDPMScheduler calculates the alpha and sigma noise schedules (based on the alpha bars) for us
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    # Initialize the DDIM ODE solver for distillation.

    if args.no_scale_pred_x0:
        use_scale = False
    else:
        use_scale = model_config["params"]["use_scale"]

    assert not use_scale
    scale_b = model_config["params"]["scale_b"]
    solver = DDIMSolver(
        noise_scheduler.alphas_cumprod.numpy(),
        ddim_timesteps=args.num_ddim_timesteps,
        use_scale=use_scale,
        scale_b=scale_b,
        ddim_eta=args.ddim_eta,
    )

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to("cpu", dtype=weight_dtype)
    text_encoder.to("cpu", dtype=weight_dtype)
    if (
        accelerator.process_index in args.reward_train_processes
        and args.reward_scale > 0
    ) or (
        accelerator.process_index in args.video_rm_train_processes
        and args.video_reward_scale > 0
    ):
        vae.to(device, weight_dtype)

    if not accelerator.process_index in args.vlcd_processes and args.use_target_unet:
        del target_unet

    # Also move the alpha and sigma noise schedules to device.
    alpha_schedule = alpha_schedule.to(device)
    sigma_schedule = sigma_schedule.to(device)
    solver = solver.to(device)

    # 10. Handle saving and loading of checkpoints
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                unet_ = deepcopy(accelerator.unwrap_model(unet))
                save_dir = os.path.join(output_dir, "unet.pt")
                torch.save(unet_.state_dict(), save_dir)
                if args.use_target_unet:
                    torch.save(
                        target_unet.state_dict(),
                        os.path.join(output_dir, "target_unet.pt"),
                    )
                for model in models:
                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()
                del unet_

        def load_model_hook(models, input_dir):
            pass

        accelerator.register_save_state_pre_hook(save_model_hook)
        # accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    temporal_params = []
    other_params = []
    named_modules_dict = dict(unet.named_modules())
    for n, p in unet.named_parameters():
        if n.startswith("init_attn.0"):
            temporal_params.append(p)
        elif len(n.split(".")) > 2:
            module_name = ".".join(n.split(".")[:3])
            if module_name in named_modules_dict and isinstance(
                named_modules_dict[module_name], TemporalTransformer
            ):
                temporal_params.append(p)
            else:
                other_params.append(p)
        else:
            other_params.append(p)

    if args.train_text_encoder:
        text_encoder.requires_grad_(True)
        clip_model = text_encoder.model

        locked_layers = [clip_model.token_embedding]
        clip_model.positional_embedding.requires_grad = False
        locked_layers.append(
            clip_model.transformer.resblocks[: -args.unlocked_text_layers]
        )
        for module in locked_layers:
            for n, p in module.named_parameters():
                p.requires_grad = False

        text_encoder.to(device, torch.float32)
        other_params += [p for p in text_encoder.parameters() if p.requires_grad]

    # 12. Optimizer creation
    optimizer = optimizer_class(
        [
            {"params": other_params},
            {
                "params": temporal_params,
                "lr": args.learning_rate * args.temporal_lr_scale,
            },
        ],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    dataset = MP4Dataset(
        args.train_shards_path_or_url, latent_root=args.latent_root
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size * accelerator.num_processes,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    train_dataloader.num_batches = len(train_dataloader)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # 15. Prepare for training
    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(
            args.tracker_project_name,
            config=tracker_config,
            init_kwargs={"wandb": {"name": args.output_dir.split("/")[-1]}},
        )

    # 16. Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    bsz = args.train_batch_size
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                if (
                    accelerator.process_index in args.video_rm_train_processes
                    and args.video_reward_scale > 0
                    and not accelerator.process_index in args.vlcd_processes
                ):
                    rbsz = args.video_rm_train_bsz

                    short_txt = batch.get("short_txt", [""] * args.train_batch_size)
                    short_txt_idx = torch.arange(args.train_batch_size)[
                        torch.tensor([t != "" for t in short_txt])
                    ]
                    num_short_txt = len(short_txt_idx)
                    if num_short_txt == 0:
                        b_idx = torch.randperm(args.train_batch_size)[:rbsz]
                    else:
                        rbsz = min(num_short_txt, rbsz)
                        b_idx = torch.randperm(num_short_txt)[:rbsz]
                        b_idx = short_txt_idx[b_idx]

                    bsz = rbsz
                    batch["txt"] = [batch["txt"][idx] for idx in b_idx]
                    batch["short_txt"] = [batch["short_txt"][idx] for idx in b_idx]
                    for k, v in batch.items():
                        if k in ["txt", "short_txt"]:
                            continue
                        batch[k] = v[b_idx]

                # 1. Load and process the image and text conditioning
                text = batch["txt"]
                index = batch["index"].to(device).long()
                noisy_model_input = batch["z_t"].to(device, weight_dtype)
                cond_teacher_output = batch["cond_teacher_out"].to(device, weight_dtype)
                uncond_teacher_output = batch["uncond_teacher_out"].to(
                    device, weight_dtype
                )
                score = batch["score"].to(device, weight_dtype)
                use_motion_guide = batch["use_motion_guide"].to(device)

                prompt_emb = batch["prompt_emb"].to(device, weight_dtype)

                index_reshape = index.reshape(bsz, 1, 1, 1, 1)
                if args.train_text_encoder:
                    encoded_text = text_encoder(text).to(device, weight_dtype)
                else:
                    encoded_text = prompt_emb

                # 2. Sample a random timestep for each image t_n from the ODE solver timesteps without bias.
                # For the DDIM solver, the timestep schedule is [T - 1, T - k - 1, T - 2 * k - 1, ...]
                start_timesteps = solver.ddim_timesteps[index]
                timesteps = start_timesteps - args.topk
                timesteps = torch.where(
                    timesteps < 0, torch.zeros_like(timesteps), timesteps
                )

                # 3. Get boundary scalings for start_timesteps and (end) timesteps.
                c_skip_start, c_out_start = scalings_for_boundary_conditions(
                    start_timesteps, timestep_scaling=args.timestep_scaling_factor
                )
                c_skip_start, c_out_start = [
                    append_dims(x, noisy_model_input.ndim)
                    for x in [c_skip_start, c_out_start]
                ]
                c_skip, c_out = scalings_for_boundary_conditions(
                    timesteps, timestep_scaling=args.timestep_scaling_factor
                )
                c_skip, c_out = [
                    append_dims(x, noisy_model_input.ndim) for x in [c_skip, c_out]
                ]

                # 5. Sample a random guidance scale w from U[w_min, w_max] and embed it
                w = (args.w_max - args.w_min) * torch.rand((bsz,)) + args.w_min
                w_embedding = guidance_scale_embedding(w, time_cond_proj_dim).to(
                    device, weight_dtype
                )
                w = w.reshape(bsz, 1, 1, 1, 1).to(device, weight_dtype)

                motion_gs = args.motion_gs * torch.ones((bsz,))
                assert use_motion_guide.shape == motion_gs.shape, (
                    use_motion_guide.shape,
                    motion_gs.shape,
                )
                if accelerator.process_index in args.vlcd_processes:
                    condition = torch.logical_and(
                        use_motion_guide,
                        index >= (1 - args.percentage) * args.num_ddim_timesteps,
                    ).to(motion_gs.device)
                    motion_gs = torch.where(
                        condition, motion_gs, torch.zeros_like(motion_gs)
                    )

                if args.use_motion_cond:
                    motion_gs_embedding = guidance_scale_embedding(
                        motion_gs, time_cond_proj_dim
                    ).to(device, weight_dtype)
                else:
                    motion_gs_embedding = None

                motion_gs = motion_gs.reshape(bsz, 1, 1, 1, 1).to(device, weight_dtype)

                # 7. Get online LCM prediction on z_{t_{n + k}} (noisy_model_input), w, c, t_{n + k} (start_timesteps)
                context = {"context": encoded_text, "fps": args.fps}
                noise_pred = unet(
                    noisy_model_input,
                    start_timesteps,
                    **context,
                    timestep_cond=w_embedding,
                    motion_cond=motion_gs_embedding,
                )
                pred_x_0 = get_predicted_original_sample(
                    noise_pred,
                    start_timesteps,
                    noisy_model_input,
                    "epsilon",
                    alpha_schedule,
                    sigma_schedule,
                )

                model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0

                distill_loss = torch.zeros_like(model_pred).mean()
                reward_loss = torch.zeros_like(model_pred).mean()
                video_rm_loss = torch.zeros_like(model_pred).mean()
                if (
                    accelerator.process_index in args.reward_train_processes
                    and args.reward_scale > 0
                ):
                    # sample args.reward_frame_bsz frames
                    compute_rm_loss = True
                    idx = torch.randperm(args.n_frames)[: args.reward_frame_bsz]
                    num_total_frames = args.reward_frame_bsz
                    short_txt = batch.get("short_txt", [""] * bsz)
                    short_txt_idx = torch.arange(bsz)[
                        torch.tensor([t != "" for t in short_txt])
                    ]
                    num_short_txt = len(short_txt_idx)
                    if num_short_txt == 0:
                        b_idx = torch.randperm(bsz)[: args.reward_train_bsz]
                        selected_text = [text[idx] for idx in b_idx]
                        num_total_frames *= args.reward_train_bsz
                        if "webvid" in args.train_shards_path_or_url and (
                            accelerator.process_index in args.vlcd_processes
                            or args.video_rm_train_processes
                        ):
                            compute_rm_loss = False
                    else:
                        reward_train_bsz = min(num_short_txt, args.reward_train_bsz)
                        b_idx = torch.randperm(num_short_txt)[:reward_train_bsz]
                        b_idx = short_txt_idx[b_idx]
                        selected_text = [short_txt[idx] for idx in b_idx]
                        num_total_frames *= reward_train_bsz

                    if compute_rm_loss:
                        selected_latents = (
                            model_pred[b_idx][:, :, idx] / vae_scale_factor
                        )
                        selected_latents = selected_latents.permute(0, 2, 1, 3, 4)
                        selected_latents = selected_latents.reshape(
                            num_total_frames, *selected_latents.shape[2:]
                        )
                        decoded_imgs = vae.decode(selected_latents.to(vae.dtype))
                        decoded_imgs = (decoded_imgs / 2 + 0.5).clamp(0, 1)
                        expert_rewards = reward_fn(decoded_imgs, selected_text)
                        reward_loss = -expert_rewards.mean() * args.reward_scale
                if (
                    accelerator.process_index in args.video_rm_train_processes
                    and args.video_reward_scale > 0
                ):
                    assert args.video_rm_train_bsz == 1
                    assert args.n_frames > args.video_rm_frame_bsz

                    skip_frames = args.n_frames // args.video_rm_frame_bsz
                    start_id = torch.randint(0, skip_frames, (1,))[0].item()
                    idx = torch.arange(start_id, args.n_frames, skip_frames)[
                        : args.video_rm_frame_bsz
                    ]
                    num_total_frames = len(idx)
                    assert len(idx) == args.video_rm_frame_bsz

                    short_txt = batch.get("short_txt", [""] * bsz)
                    short_txt_idx = torch.arange(bsz)[
                        torch.tensor([t != "" for t in short_txt])
                    ]
                    num_short_txt = len(short_txt_idx)
                    compute_video_rm_loss = True
                    if num_short_txt == 0:
                        b_idx = torch.randperm(bsz)[: args.video_rm_train_bsz]
                        selected_text = [text[idx] for idx in b_idx]
                        num_total_frames *= args.reward_train_bsz
                        if "webvid" in args.train_shards_path_or_url and (
                            accelerator.process_index in args.vlcd_processes
                            or accelerator.process_index in args.reward_train_processes
                        ):
                            compute_video_rm_loss = False
                    else:
                        reward_train_bsz = min(num_short_txt, args.video_rm_train_bsz)
                        b_idx = torch.randperm(num_short_txt)[:reward_train_bsz]
                        b_idx = short_txt_idx[b_idx]
                        selected_text = [short_txt[idx] for idx in b_idx]
                        num_total_frames *= reward_train_bsz

                    if compute_video_rm_loss:
                        selected_latents = (
                            model_pred[b_idx][:, :, idx] / vae_scale_factor
                        )
                        selected_latents = selected_latents.permute(0, 2, 1, 3, 4)
                        selected_latents = selected_latents.reshape(
                            num_total_frames, *selected_latents.shape[2:]
                        )
                        decoded_imgs = vae.decode(selected_latents.to(vae.dtype))
                        decoded_imgs = (decoded_imgs / 2 + 0.5).clamp(0, 1)
                        decoded_imgs = decoded_imgs.reshape(
                            bsz, args.video_rm_frame_bsz, *decoded_imgs.shape[1:]
                        )
                        video_rewards = video_rm_fn(decoded_imgs, text)
                        video_rm_loss = -video_rewards.mean() * args.video_reward_scale

                if accelerator.process_index in args.vlcd_processes:
                    # 8. Compute the conditional and unconditional teacher model predictions to get CFG estimates of the
                    # predicted noise eps_0 and predicted original sample x_0, then run the ODE solver using these
                    # estimates to predict the data point in the augmented PF-ODE trajectory corresponding to the next ODE
                    # solver timestep.
                    with torch.no_grad():
                        with torch.autocast("cuda"):
                            # 1. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and conditional embedding c
                            cond_pred_x0 = get_predicted_original_sample(
                                cond_teacher_output,
                                start_timesteps,
                                noisy_model_input,
                                "epsilon",
                                alpha_schedule,
                                sigma_schedule,
                            )
                            cond_pred_noise = get_predicted_noise(
                                cond_teacher_output,
                                start_timesteps,
                                noisy_model_input,
                                "epsilon",
                                alpha_schedule,
                                sigma_schedule,
                            )

                            # 2. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and unconditional embedding 0
                            uncond_pred_x0 = get_predicted_original_sample(
                                uncond_teacher_output,
                                start_timesteps,
                                noisy_model_input,
                                "epsilon",
                                alpha_schedule,
                                sigma_schedule,
                            )
                            uncond_pred_noise = get_predicted_noise(
                                uncond_teacher_output,
                                start_timesteps,
                                noisy_model_input,
                                "epsilon",
                                alpha_schedule,
                                sigma_schedule,
                            )

                            # 3. Calculate the CFG estimate of x_0 (pred_x0) and eps_0 (pred_noise)
                            # Note that this uses the LCM paper's CFG formulation rather than the Imagen CFG formulation
                            pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                            pred_noise = cond_pred_noise + w * (
                                cond_pred_noise - uncond_pred_noise
                            )

                            # Motion Prior Guidance
                            alphas = extract_into_tensor(
                                alpha_schedule, start_timesteps, score.shape
                            )
                            condition = torch.logical_and(
                                use_motion_guide.reshape(bsz, 1, 1, 1, 1),
                                index_reshape
                                >= (1 - args.percentage) * args.num_ddim_timesteps,
                            )
                            alphas = torch.where(
                                condition, alphas, torch.ones_like(alphas)
                            )
                            pred_noise -= motion_gs * (1 - alphas) ** (0.5) * score

                            # 4. Run one step of the ODE solver to estimate the next point x_prev on the
                            # augmented PF-ODE trajectory (solving backward in time)
                            # Note that the DDIM step depends on both the predicted x_0 and source noise eps_0.
                            x_prev = solver.ddim_step(pred_x0, pred_noise, index).to(
                                weight_dtype
                            )

                    # 9. Get target LCM prediction on x_prev, w, c, t_n (timesteps)
                    with torch.no_grad():
                        with torch.autocast("cuda", dtype=weight_dtype):
                            unet_fn = target_unet if args.use_target_unet else unet
                            target_noise_pred = unet_fn(
                                x_prev,
                                timesteps,
                                **context,
                                timestep_cond=w_embedding,
                                motion_cond=motion_gs_embedding,
                            )
                        pred_x_0 = get_predicted_original_sample(
                            target_noise_pred.to(torch.float32),
                            timesteps,
                            x_prev,
                            "epsilon",
                            alpha_schedule,
                            sigma_schedule,
                        )
                        target = c_skip * x_prev + c_out * pred_x_0

                    # 10. Calculate loss
                    if args.loss_type == "l2":
                        distill_loss = F.mse_loss(model_pred, target, reduction="mean")
                    elif args.loss_type == "huber":
                        distill_loss = huber_loss(model_pred, target, args.huber_c)

                accelerator.backward(distill_loss + reward_loss + video_rm_loss)
                # 11. Backpropagate on the online student model (`unet`)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # 12. Make EMA update to target student model parameters (`target_unet`)
                if args.use_target_unet:
                    update_ema(
                        target_unet.parameters(), unet.parameters(), args.ema_decay
                    )
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d
                                for d in checkpoints
                                if d.startswith("checkpoint") and not "rm" in d
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if (
                        global_step % args.validation_steps == 0
                        and args.report_to == "wandb"
                    ):
                        log_validation(
                            pretrained_t2v,
                            unet,
                            noise_scheduler,
                            model_config,
                            args,
                            accelerator,
                        )

                # Gather losses from all processes
                distill_loss_list = accelerator.gather(distill_loss.detach())
                reward_loss_list = accelerator.gather(reward_loss.detach().float())
                video_rm_loss_list = accelerator.gather(video_rm_loss.detach().float())
                if accelerator.is_main_process:
                    distill_loss = distill_loss_list.sum() / len(args.vlcd_processes)
                    reward_loss = (
                        reward_loss_list.sum()
                        / len(args.reward_train_processes)
                        / args.reward_scale
                    )
                    video_rm_loss = (
                        video_rm_loss_list.sum()
                        / len(args.video_rm_train_processes)
                        / args.video_reward_scale
                    )
                    logs = {
                        "distillation loss": distill_loss.detach().item(),
                        "image reward loss": reward_loss.detach().item(),
                        "video reward loss": video_rm_loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    }

                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)
                    del distill_loss, reward_loss, video_rm_loss, model_pred
                    gc.collect()

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)

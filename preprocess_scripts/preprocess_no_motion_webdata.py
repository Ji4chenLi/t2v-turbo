import argparse
import logging
import math
import os
import pickle
import random
from pathlib import Path
import boto3

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.utils.checkpoint
from tqdm import tqdm
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers

from webdataset import WebLoader
from data.dataset import get_video_dataset

from motion_prior_sample import get_motion_prior_score, reverse_ddim_loop
from ode_solver import DDIMSolver
from scheduler.t2v_turbo_scheduler import T2VTurboScheduler
from utils.common_utils import load_model_checkpoint
from utils.utils import instantiate_from_config


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # ---------- AWS S3 Arguments ----------
    parser.add_argument(
        "--s3_bucket_name",
        type=str,
        default="BUCKET_NAME",
        help="The name of the S3 bucket.",
    )
    # ----------Model Checkpoint Loading Arguments----------
    parser.add_argument(
        "--pretrained_model_cfg",
        type=str,
        default="configs/inference_t2v_512_v2.0_motion_clone.yaml",
        help="Pretrained Model Config.",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="model_cache/VideoCrafter2_model.ckpt",
        help="Path to the pretrained model.",
    )
    # ----General Training Arguments----
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/process_webdata",
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
    # ----Image Processing----
    parser.add_argument(
        "--shards_path_or_url",
        type=str,
        default="pipe:aws s3 cp s3://path/to/webvid_data/{00000..00399}.tar -",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="path/to/save_root",
        help="The path to the root directory of the save files in a S3 bucket.",
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
        "--batch_size",
        type=int,
        default=3,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=240000,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=16,
        help="Number of frames to sample from a video.",
    )
    # ----Learning Rate----
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
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt for the DDIM sampling.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="1000 (Num Train timesteps) // 50 (Num timesteps for DDIM sampling)",
    )
    # ----Distributed Training----
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


# Adapted from pipelines.StableDiffusionPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoder, is_train=True):
    captions = []
    for caption in prompt_batch:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        prompt_embeds = text_encoder(prompt_batch)

    return prompt_embeds


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with="tensorboard",
        project_config=accelerator_project_config,
        split_batches=True,
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

    device = accelerator.device
    weight_dtype = torch.bfloat16

    # 5. Load teacher Model
    config = OmegaConf.load(args.pretrained_model_cfg)
    model_config = config.pop("model", OmegaConf.create())
    model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False
    pretrained_t2v = instantiate_from_config(model_config)
    pretrained_t2v = load_model_checkpoint(
        pretrained_t2v,
        args.pretrained_model_path,
    )
    assert not pretrained_t2v.scale_by_std

    unet = pretrained_t2v.model.diffusion_model.to(device, weight_dtype)
    vae = pretrained_t2v.first_stage_model.to(device, weight_dtype)
    vae_scale_factor = model_config["params"]["scale_factor"]
    text_encoder = pretrained_t2v.cond_stage_model.to(device, weight_dtype)

    unet.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    text_encoder.requires_grad_(False).eval()

    # 1. Create the noise scheduler and the desired noise schedule.
    noise_scheduler = T2VTurboScheduler(
        linear_start=model_config["params"]["linear_start"],
        linear_end=model_config["params"]["linear_end"],
    )

    solver = DDIMSolver(
        noise_scheduler.alphas_cumprod.numpy(),
        ddim_timesteps=args.num_ddim_timesteps,
        use_scale=False,
    ).to(device, weight_dtype)

    # Also move the alpha and sigma noise schedules to accelerator.device.
    solver = solver.to(accelerator.device)

    torch.backends.cuda.matmul.allow_tf32 = True

    decoder_kwargs = {
        "n_frames": args.n_frames,  # get 16 frames from each video
        "fps": args.fps,
        "num_threads": 12,  # use 16 threads to decode the video
    }
    resolution = tuple([s * 8 for s in model_config["params"]["image_size"]])
    dataset = get_video_dataset(
        urls=args.shards_path_or_url,
        batch_size=args.batch_size,
        decoder_kwargs=decoder_kwargs,
        resize_size=resolution,
        crop_size=resolution,
    )
    num_workers = args.dataloader_num_workers
    train_dataloader = WebLoader(dataset, batch_size=None, num_workers=num_workers)
    global_batch_size = args.batch_size * accelerator.num_processes
    num_worker_batches = math.ceil(args.max_samples / (global_batch_size * num_workers))

    train_dataloader.num_batches = num_worker_batches * args.dataloader_num_workers
    train_dataloader.num_samples = train_dataloader.num_batches * global_batch_size

    # 15. Prepare for training
    # Prepare everything with our `accelerator`.
    uncond_prompt_embeds = text_encoder([args.negative_prompt] * args.batch_size).to(
        device, weight_dtype
    )
    uncond_context = {"context": torch.cat([uncond_prompt_embeds], 1), "fps": args.fps}

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(
            "t2v-turbo-v2",
            config=tracker_config,
            init_kwargs={"wandb": {"name": args.output_dir.split("/")[-1]}},
        )

    s3 = boto3.client("s3")

    progress_bar = tqdm(
        range(0, train_dataloader.num_batches),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for batch in train_dataloader:
        # 1. Load and process the image and text conditioning
        video = batch["mp4"]
        text = batch["txt"]
        video_ids = [item["videoid"] for item in batch["json"]]
        video = ((video / 255.0).clamp(0.0, 1.0) - 0.5) / 0.5
        video = video.permute(0, 1, 4, 2, 3)
        video = video.to(accelerator.device, weight_dtype, non_blocking=True)

        b, t = video.shape[:2]
        pixel_values_flatten = video.view(b * t, *video.shape[2:])

        with torch.no_grad():
            latents = vae.encode(pixel_values_flatten).sample()
            latents = latents.view(b, t, *latents.shape[1:])
            # Convert latents from (b, t, c, h, w) to (b, c, t, h, w)
            latents = latents.permute(0, 2, 1, 3, 4)
            latents = latents * vae_scale_factor

            prompt_embeds = text_encoder(text).to(device, weight_dtype)
            context = {"context": torch.cat([prompt_embeds], 1), "fps": args.fps}
            bsz = latents.shape[0]

            # 2. Sample a random timestep for each image t_n from the ODE solver timesteps without bias.
            # For the DDIM solver, the timestep schedule is [T - 1, T - k - 1, T - 2 * k - 1, ...]
            index = torch.randint(
                0, args.num_ddim_timesteps, (bsz,), device=latents.device
            ).long()
            start_timesteps = solver.ddim_timesteps[index]
            timesteps = start_timesteps - args.topk
            timesteps = torch.where(
                timesteps < 0, torch.zeros_like(timesteps), timesteps
            )

            # Sample noise from the prior and add it to the latents according to the noise magnitude at each
            # timestep (this is the forward diffusion process) [z_{t_{n + k}} in Algorithm 1]
            noise = torch.randn_like(latents)
            z_ts = noise_scheduler.add_noise(latents, noise, start_timesteps)
            cond_teacher_output = unet(z_ts, start_timesteps, **context)
            uncond_teacher_output = unet(z_ts, start_timesteps, **uncond_context)

        for (
            idx,
            z_t,
            cond_teacher_out,
            uncond_teacher_out,
            prompt_emb,
            video_id,
            txt,
        ) in zip(
            index,
            z_ts,
            cond_teacher_output,
            uncond_teacher_output,
            prompt_embeds,
            video_ids,
            text,
        ):
            zeros = torch.zeros_like(z_t).to(torch.float16)
            to_save = {
                "index": idx,
                "z_t": z_t.to(torch.float16),
                "cond_teacher_out": cond_teacher_out.to(torch.float16),
                "uncond_teacher_out": uncond_teacher_out.to(torch.float16),
                "score": zeros,
                "z_example": zeros,
                "z_example_prev": zeros,
                "prompt_emb": prompt_emb.to(torch.float16),
            }
            to_save = {k: v.detach().cpu() for k, v in to_save.items()}
            to_save["text"] = txt
            to_save = pickle.dumps(to_save)
            s3.put_object(
                Bucket=args.s3_bucket_name,
                Key=f"{args.save_root}/{video_id}.pkl",
                Body=to_save,
            )
        progress_bar.update(1)

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)

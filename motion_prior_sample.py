import argparse
from copy import deepcopy
import os
import pickle
from omegaconf import OmegaConf
import torch
import torchvision
from tqdm import tqdm
from ode_solver.ddim_solver import DDIMSolver
from scheduler.t2v_turbo_scheduler import T2VTurboScheduler
from utils.utils import instantiate_from_config
from utils.common_utils import (
    extract_into_tensor,
    get_transform,
    get_predicted_noise,
    get_predicted_original_sample,
    load_model_checkpoint,
    read_video_to_tensor,
    compute_temp_loss,
)
from reward_fn import get_reward_fn


SAMPLE_SIZE = (320, 512)


def reverse_ddim_loop(latents, unet, context, solver, num_inference_steps, device):
    intermediate_latents = []
    for i in range(num_inference_steps):
        index = torch.full((1,), i, device=device, dtype=torch.long)
        ts = solver.ddim_timesteps[index].long()
        pred_noise = unet(latents, ts, **context)

        latents = solver.ddim_reverse_step(latents, pred_noise, ts)
        intermediate_latents.append(latents)

    return intermediate_latents


def get_temp_attn_prob(unet, latent, ts, context):
    attention_prob = {}
    model_output = unet(latent, ts, **context)
    for name, module in unet.named_modules():
        if (
            name.startswith("output_blocks.3.2")
            or name.startswith("output_blocks.4.2")
            or name.startswith("output_blocks.5.2")
            or name.startswith("output_blocks.6.2")
            or name.startswith("output_blocks.7.2")
            or name.startswith("output_blocks.8.2")
            or name.startswith("output_blocks.9.2")
            or name.startswith("output_blocks.10.2")
            or name.startswith("output_blocks.11.2")
        ) and name.endswith("blocks.0.attn1"):
            attention_prob[name] = module.attention_probs
    return model_output, attention_prob


def get_motion_prior_score(
    unet,
    latents,
    ts,
    example_latent,
    original_context,
    inference_context,
    temp_loss_scale,
):
    attention_prob = {}
    attention_prob_example = {}
    with torch.no_grad():
        _, attention_prob_example = get_temp_attn_prob(
            unet, example_latent, ts, original_context
        )
    with torch.set_grad_enabled(True):
        latents.requires_grad_(True)
        cond_teacher_output, attention_prob = get_temp_attn_prob(
            unet, latents, ts, inference_context
        )
        loss = temp_loss_scale * compute_temp_loss(
            attention_prob, attention_prob_example
        )
        score = torch.autograd.grad(loss, latents)[0].detach()

    return score, cond_teacher_output


@torch.no_grad()
def main(
    unet,
    vae,
    text_encoder,
    scheduler,
    pretrained_t2v,
    solver,
    reward_fn,
    args,
    device,
    dtype,
):
    pixel_transforms = get_transform(sample_size=SAMPLE_SIZE)

    video = read_video_to_tensor(
        args.ref_video_path, sample_fps=args.fps, sample_frames=args.n_frames
    )
    video = pixel_transforms(video)

    latents = vae.encode(video.to(device, dtype)).sample()
    latents = latents.unsqueeze(0)
    latents = latents.permute(0, 2, 1, 3, 4)
    vae_scale_factor = 0.18215
    latents = latents * vae_scale_factor

    inference_prompt_embeds = text_encoder([args.inference_prompt]).to(device, dtype)
    original_prompt_embeds = text_encoder([args.ref_prompt]).to(device, dtype)
    uncond_prompt_embeds = text_encoder([""]).to(device, dtype)

    inference_context = {
        "context": torch.cat([inference_prompt_embeds], 1),
        "fps": args.fps,
    }
    original_context = {
        "context": torch.cat([original_prompt_embeds], 1),
        "fps": args.fps,
    }
    uncond_context = {"context": torch.cat([uncond_prompt_embeds], 1), "fps": args.fps}

    alpha_schedule = torch.sqrt(scheduler.alphas_cumprod).to(device)
    sigma_schedule = torch.sqrt(1 - scheduler.alphas_cumprod).to(device)

    # Reversed DDIM
    unet.to(dtype)
    num_inference_steps = len(solver.ddim_timesteps)
    original_latents = latents.to(dtype)
    if not os.path.exists(args.intermediate_latents_path):
        print("Starting reversed DDIM...")
        latents = deepcopy(original_latents)
        intermediate_latents = reverse_ddim_loop(
            latents, unet, original_context, solver, num_inference_steps, device
        )
        print("Finished reversed DDIM.")
        with open("resources/intermediate_latents.pkl", "wb") as f:
            pickle.dump(intermediate_latents, f)
    else:
        with open(args.intermediate_latents_path, "rb") as f:
            intermediate_latents = pickle.load(f)
    # DDIM
    # latents = torch.randn_like(intermediate_latents[-1])
    noise = torch.randn_like(intermediate_latents[-1])
    latents = scheduler.add_noise(original_latents, noise, solver.ddim_timesteps[-1])
    if not os.path.exists(args.motion_intermediate_path):
        motion_intermediate = []
    else:
        with open(args.motion_intermediate_path, "rb") as f:
            motion_intermediate = pickle.load(f)

    sampled_latents = []
    for i in tqdm(range(num_inference_steps - 1, -1, -1)):
        index = torch.full((1,), i, device=device, dtype=torch.long)
        ts = solver.ddim_timesteps[index]
        if i > num_inference_steps - args.percentage * num_inference_steps:
            idx = num_inference_steps - 1 - i
            if len(motion_intermediate) > idx:
                results = motion_intermediate[idx]
                cond_pred_x_0 = results["cond_pred_x_0"].to(device, dtype)
                cond_pred_noise = results["cond_pred_noise"].to(device, dtype)
                uncond_pred_x_0 = results["uncond_pred_x_0"].to(device, dtype)
                uncond_pred_noise = results["uncond_pred_noise"].to(device, dtype)
                score = results["score"].to(device, dtype)
            else:
                score, cond_pred_noise = get_motion_prior_score(
                    unet,
                    latents,
                    ts,
                    intermediate_latents[i],
                    original_context,
                    inference_context,
                    args.temp_loss_scale,
                )
                with torch.set_grad_enabled(False):
                    cond_pred_x_0 = get_predicted_original_sample(
                        cond_pred_noise,
                        ts,
                        latents,
                        "epsilon",
                        alpha_schedule,
                        sigma_schedule,
                    )
                    uncond_pred_noise = unet(latents, ts, **uncond_context)
                    uncond_pred_x_0 = get_predicted_original_sample(
                        uncond_pred_noise,
                        ts,
                        latents,
                        "epsilon",
                        alpha_schedule,
                        sigma_schedule,
                    )
                motion_intermediate.append(
                    {
                        "cond_pred_x_0": cond_pred_x_0.detach().cpu(),
                        "uncond_pred_x_0": uncond_pred_x_0.detach().cpu(),
                        "cond_pred_noise": cond_pred_noise.detach().cpu(),
                        "uncond_pred_noise": uncond_pred_noise.detach().cpu(),
                        "score": score.detach().cpu(),
                    }
                )

            if args.use_rm_guide and args.reward_scale > 0.0:
                with torch.set_grad_enabled(False):
                    cond_pred_noise = unet(latents, ts, **inference_context)
                    uncond_pred_noise = unet(latents, ts, **uncond_context)
                    pred_noise = cond_pred_noise + args.guidance_scale * (
                        cond_pred_noise - uncond_pred_noise
                    )
                    cond_pred_x_0 = get_predicted_original_sample(
                        cond_pred_noise,
                        ts,
                        latents,
                        "epsilon",
                        alpha_schedule,
                        sigma_schedule,
                    )
                    uncond_pred_x_0 = get_predicted_original_sample(
                        uncond_pred_noise,
                        ts,
                        latents,
                        "epsilon",
                        alpha_schedule,
                        sigma_schedule,
                    )

                with torch.set_grad_enabled(True):
                    latents.requires_grad_(True)
                    pred_x_0 = get_predicted_original_sample(
                        pred_noise,
                        ts,
                        latents,
                        "epsilon",
                        alpha_schedule,
                        sigma_schedule,
                    )

                    idx = torch.randint(0, args.n_frames, (args.reward_batch_size,))

                    selected_latents = (
                        pred_x_0[:, :, idx].to(vae.dtype) / vae_scale_factor
                    )
                    selected_latents = selected_latents.permute(0, 2, 1, 3, 4)
                    selected_latents = selected_latents.reshape(
                        args.reward_batch_size, *selected_latents.shape[2:]
                    )
                    decoded_imgs = vae.decode(selected_latents)
                    decoded_imgs = (decoded_imgs / 2 + 0.5).clamp(0, 1)
                    expert_rewards = reward_fn(decoded_imgs, args.inference_prompt)
                    reward_loss = -expert_rewards.mean() * args.reward_scale
                    score += torch.autograd.grad(reward_loss, latents)[0].detach()
            else:
                score += torch.zeros_like(latents)
        else:
            with torch.set_grad_enabled(False):
                cond_pred_noise = unet(latents, ts, **inference_context)
                uncond_pred_noise = unet(latents, ts, **uncond_context)
                pred_noise = cond_pred_noise + args.guidance_scale * (
                    cond_pred_noise - uncond_pred_noise
                )
                cond_pred_x_0 = get_predicted_original_sample(
                    cond_pred_noise,
                    ts,
                    latents,
                    "epsilon",
                    alpha_schedule,
                    sigma_schedule,
                )
                uncond_pred_x_0 = get_predicted_original_sample(
                    uncond_pred_noise,
                    ts,
                    latents,
                    "epsilon",
                    alpha_schedule,
                    sigma_schedule,
                )
            score = torch.zeros_like(latents)

        with torch.set_grad_enabled(False):
            pred_x_0 = cond_pred_x_0 + args.guidance_scale * (
                cond_pred_x_0 - uncond_pred_x_0
            )
            pred_noise = cond_pred_noise + args.guidance_scale * (
                cond_pred_noise - uncond_pred_noise
            )
            alphas = extract_into_tensor(alpha_schedule, ts, score.shape)
            pred_noise -= (1 - alphas) ** (0.5) * score
            latents = solver.ddim_step(pred_x_0, pred_noise, index).to(device, dtype)

        sampled_latents.append(pred_x_0)

    with open(args.motion_intermediate_path, "wb") as f:
        pickle.dump(motion_intermediate, f)

    videos = pretrained_t2v.decode_first_stage_2DAE(latents)
    videos = (videos.clamp(-1.0, 1.0) + 1.0) / 2.0
    videos = (videos * 255).to(torch.uint8).permute(0, 2, 1, 3, 4).cpu().numpy()

    torchvision.io.write_video(
        args.output_video_path,
        torch.from_numpy(videos)[0].permute(0, 2, 3, 1),
        fps=args.fps,
        video_codec="h264",
        options={"crf": "10"},
    )

    if args.save_intermediate_videos:
        os.makedirs("motion_intermediates", exist_ok=True)
        for i, latents in enumerate(sampled_latents):
            videos = pretrained_t2v.decode_first_stage_2DAE(latents)
            videos = (videos.clamp(-1.0, 1.0) + 1.0) / 2.0
            videos = (videos * 255).to(torch.uint8).permute(0, 2, 1, 3, 4).cpu().numpy()

            torchvision.io.write_video(
                args.output_video_path.replace(".mp4", f"_{i}.mp4").replace(
                    "samples/", "motion_intermediates/"
                ),
                torch.from_numpy(videos)[0].permute(0, 2, 3, 1),
                fps=args.fps,
                video_codec="h264",
                options={"crf": "10"},
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motion Clone Sampling")
    # Add model name as parameter
    parser.add_argument(
        "--sampling_steps", type=int, default=200, help="Number of sampling steps"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5, help="Guidance scale"
    )
    parser.add_argument("--n_frames", type=int, default=16, help="Number of frames")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument(
        "--percentage",
        type=float,
        default=0.6,
        help="Percentage of steps to apply guidance",
    )
    parser.add_argument(
        "--temp_loss_scale",
        type=float,
        default=100,
        help="Temperature loss scale",
    )
    parser.add_argument(
        "--ref_video_path",
        type=str,
        default="resources/sample_fox.mp4",
        help="Path to reference video",
    )
    parser.add_argument(
        "--inference_prompt",
        type=str,
        default="A cat turning its heads in the bedroom.",
        help="Inference prompt for the video",
    )
    parser.add_argument(
        "--ref_prompt",
        type=str,
        default="A fox turning its heads in the woods.",
        help="Inference prompt for the video",
    )
    parser.add_argument(
        "--output_video_path",
        type=str,
        default="samples/motion_prior_sample_video.mp4",
        help="Path to output video",
    )
    parser.add_argument(
        "--sample_config_path",
        type=str,
        default="configs/inference_t2v_512_v2.0_fp16.yaml",
        help="Path to output video",
    )
    parser.add_argument(
        "--vc2_model_path",
        type=str,
        default="model_cache/VideoCrafter2_model.ckpt",
        help="Path to output video",
    )
    parser.add_argument(
        "--use_rm_guide",
        action="store_true",
        default=False,
        help="Use reward model to guide the generation",
    )
    parser.add_argument(
        "--reward_batch_size",
        type=int,
        default=12,
        help="Batch size (per device) for optimizing the text-image RM.",
    )
    parser.add_argument(
        "--reward_scale",
        type=float,
        default=1.0,
        help="The scale of the reward loss",
    )
    parser.add_argument(
        "--intermediate_latents_path",
        type=str,
        default="resources/intermediate_latents.pkl",
        help="Path to intermediate latents",
    )
    parser.add_argument(
        "--motion_intermediate_path",
        type=str,
        default="resources/motion_intermediate.pkl",
        help="Path to intermediate motion guidance results",
    )
    parser.add_argument(
        "--save_intermediate_videos",
        action="store_true",
        default=False,
        help="Save intermediate videos",
    )

    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.float16
    config = OmegaConf.load(args.sample_config_path)
    model_config = config.pop("model", OmegaConf.create())
    pretrained_t2v = instantiate_from_config(model_config)
    pretrained_t2v = load_model_checkpoint(pretrained_t2v, args.vc2_model_path)
    unet = pretrained_t2v.model.diffusion_model.to(device, dtype)
    vae = pretrained_t2v.first_stage_model.to(device, dtype)
    text_encoder = pretrained_t2v.cond_stage_model.to(device, dtype)

    unet.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    text_encoder.requires_grad_(False).eval()

    scheduler = T2VTurboScheduler(
        linear_start=model_config["params"]["linear_start"],
        linear_end=model_config["params"]["linear_end"],
    )

    solver = DDIMSolver(
        scheduler.alphas_cumprod.numpy(),
        ddim_timesteps=args.sampling_steps,
        use_scale=False,
    ).to(device, dtype)

    if args.use_rm_guide:
        reward_fn = get_reward_fn("hpsv2", precision="fp16")
    else:
        reward_fn = None

    main(
        unet,
        vae,
        text_encoder,
        scheduler,
        pretrained_t2v,
        solver,
        reward_fn,
        args,
        device,
        dtype,
    )

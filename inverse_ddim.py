from copy import deepcopy
from omegaconf import OmegaConf
import torch
import torchvision

from tqdm import tqdm
from ode_solver.ddim_solver import DDIMSolver
from scheduler.t2v_turbo_scheduler import T2VTurboScheduler
from utils.utils import instantiate_from_config
from utils.common_utils import (
    get_transform,
    get_predicted_noise,
    get_predicted_original_sample,
    load_model_checkpoint,
    read_video_to_tensor,
)

SAMPLE_SIZE = (320, 512)

@torch.no_grad()
def main(unet, vae, text_encoder, scheduler, pretrained_t2v, solver, device, dtype):
    # DDPMScheduler calculates the alpha and sigma noise schedules (based on the alpha bars) for us
    alpha_schedule = torch.sqrt(scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - scheduler.alphas_cumprod)

    video = read_video_to_tensor(
        "assets/reference_videos/sample_fox.mp4", sample_fps=8, sample_frames=16
    )
    pixel_transforms = get_transform(sample_size=SAMPLE_SIZE)
    video = pixel_transforms(video)

    latents = vae.encode(video.to(device, dtype)).sample()
    latents = latents.unsqueeze(0)
    latents = latents.permute(0, 2, 1, 3, 4)
    latents = latents * 0.18215

    prompt_embeds = text_encoder.encode(["Flowers and grassland on the shore"])
    context = {
        "context": torch.cat([prompt_embeds.to(dtype)], 1),
        "fps": 8,
    }
    alpha_schedule = torch.sqrt(scheduler.alphas_cumprod).to(device)
    sigma_schedule = torch.sqrt(1 - scheduler.alphas_cumprod).to(device)

    # Reversed DDIM
    num_inference_steps = len(solver.ddim_timesteps)
    unet.to(dtype)
    latents = latents.to(dtype)

    intermediate_latents = []
    for i in tqdm(range(num_inference_steps), total=num_inference_steps):
        index = torch.full((1,), i, device=device, dtype=torch.long)
        ts = solver.ddim_timesteps[index].long()
        pred_noise = unet(latents, ts, **context)

        latents = solver.ddim_reverse_step(latents, pred_noise, ts).to(dtype)
        intermediate_latents.append(latents)

    videos = pretrained_t2v.decode_first_stage_2DAE(intermediate_latents[-1])
    videos = (videos.clamp(-1.0, 1.0) + 1.0) / 2.0
    videos = (videos * 255).to(torch.uint8).permute(0, 2, 1, 3, 4).cpu().numpy()

    torchvision.io.write_video(
        f"noisy_video.mp4",
        torch.from_numpy(videos)[0].permute(0, 2, 3, 1),
        fps=8,
        video_codec="h264",
        options={"crf": "10"},
    )

    # DDIM
    latents = deepcopy(intermediate_latents[-1])
    noisy_intermediate_latents = []
    for i in tqdm(range(num_inference_steps - 1, -1, -1)):
        index = torch.full((1,), i, device=device, dtype=torch.long)
        ts = solver.ddim_timesteps[index]
        # model prediction (v-prediction, eps, x)
        pred_noise = unet(latents, ts, **context)
        pred_x_0 = get_predicted_original_sample(
            pred_noise, ts, latents, "epsilon", alpha_schedule, sigma_schedule
        )

        latents = solver.ddim_step(pred_x_0, pred_noise, index).to(dtype)
        noisy_intermediate_latents.append(latents)

    videos = pretrained_t2v.decode_first_stage_2DAE(latents)
    videos = (videos.clamp(-1.0, 1.0) + 1.0) / 2.0
    videos = (videos * 255).to(torch.uint8).permute(0, 2, 1, 3, 4).cpu().numpy()

    torchvision.io.write_video(
        f"reconstructed_video.mp4",
        torch.from_numpy(videos)[0].permute(0, 2, 3, 1),
        fps=8,
        video_codec="h264",
        options={"crf": "10"},
    )


if __name__ == "__main__":
    # Add model name as parameter
    ddim_timesteps = 200
    device = torch.device("cuda")
    dtype = torch.bfloat16
    config = OmegaConf.load("configs/inference_t2v_512_v2.0_motion_clone.yaml")
    model_config = config.pop("model", OmegaConf.create())
    pretrained_t2v = instantiate_from_config(model_config)
    pretrained_t2v = load_model_checkpoint(
        pretrained_t2v, "model_cache/VideoCrafter2_model.ckpt"
    )
    unet = pretrained_t2v.model.diffusion_model.to(device, dtype)
    unet.dtype = dtype
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
        ddim_timesteps=ddim_timesteps,
        use_scale=False,
    ).to(device)

    main(unet, vae, text_encoder, scheduler, pretrained_t2v, solver, device, dtype)

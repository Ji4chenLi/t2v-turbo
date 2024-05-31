# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import time
from omegaconf import OmegaConf
from cog import BasePredictor, Input, Path
import torch
import torchvision

from utils.lora import collapse_lora, monkeypatch_remove_lora
from utils.common_utils import load_model_checkpoint
from utils.utils import instantiate_from_config
from utils.lora_handler import LoraHandler
from scheduler.t2v_turbo_scheduler import T2VTurboScheduler
from pipeline.t2v_turbo_vc2_pipeline import T2VTurboVC2Pipeline


MODEL_URL = "https://weights.replicate.delivery/default/Ji4chenLi/t2v-turbo.tar"
MODEL_CACHE = "model_cache"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        device = "cuda"

        base_model_dir = os.path.join(MODEL_CACHE, "VideoCrafter2_model.ckpt")
        unet_dir = os.path.join(MODEL_CACHE, "unet_lora.pt")

        config = OmegaConf.load("configs/inference_t2v_512_v2.0.yaml")
        model_config = config.pop("model", OmegaConf.create())
        pretrained_t2v = instantiate_from_config(model_config)
        pretrained_t2v = load_model_checkpoint(pretrained_t2v, base_model_dir)

        unet_config = model_config["params"]["unet_config"]
        unet_config["params"]["time_cond_proj_dim"] = 256
        unet = instantiate_from_config(unet_config)

        unet.load_state_dict(
            pretrained_t2v.model.diffusion_model.state_dict(), strict=False
        )

        use_unet_lora = True
        lora_manager = LoraHandler(
            version="cloneofsimo",
            use_unet_lora=use_unet_lora,
            save_for_webui=True,
            unet_replace_modules=["UNetModel"],
        )
        lora_manager.add_lora_to_model(
            use_unet_lora,
            unet,
            lora_manager.unet_replace_modules,
            lora_path=unet_dir,
            dropout=0.1,
            r=64,
        )
        unet.eval()
        collapse_lora(unet, lora_manager.unet_replace_modules)
        monkeypatch_remove_lora(unet)

        pretrained_t2v.model.diffusion_model = unet
        scheduler = T2VTurboScheduler(
            linear_start=model_config["params"]["linear_start"],
            linear_end=model_config["params"]["linear_end"],
        )
        self.pipeline = T2VTurboVC2Pipeline(pretrained_t2v, scheduler, model_config)
        self.pipeline.to(device, torch_dtype=torch.float16)

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="With the style of low-poly game art, A majestic, white horse gallops gracefully across a moonlit beach",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=8, default=4
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        num_frames: int = Input(
            description="Number of Video Frames",
            choices=[8, 16, 24, 32, 40, 48],
            default=16,
        ),
        fps: int = Input(
            description="FPS of the output video.",
            choices=[8, 12, 16, 20, 24, 28, 32],
            default=8,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        generator = torch.Generator("cuda").manual_seed(seed)

        result = self.pipeline(
            prompt=prompt,
            frames=num_frames,
            fps=fps,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_videos_per_prompt=1,
            generator=generator,
        )

        out_path = "/tmp/out.mp4"

        video = result[0].detach().cpu()
        video = torch.clamp(video.float(), -1.0, 1.0)
        video = video.permute(1, 0, 2, 3)
        video = (video + 1.0) / 2.0
        video = (video * 255).to(torch.uint8).permute(0, 2, 3, 1)

        torchvision.io.write_video(
            out_path, video, fps=fps, video_codec="h264", options={"crf": "10"}
        )

        return Path(out_path)

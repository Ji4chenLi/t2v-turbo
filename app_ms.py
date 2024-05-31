# Adapted from https://github.com/luosiallen/latent-consistency-model
from __future__ import annotations

import argparse
import os
import random
import time

import gradio as gr
import numpy as np

from scheduler.t2v_turbo_scheduler import T2VTurboScheduler
from pipeline.t2v_turbo_ms_pipeline import T2VTurboMSPipeline
from utils.common_utils import set_torch_2_attn

try:
    import intel_extension_for_pytorch as ipex
except:
    pass

from transformers import CLIPTokenizer, CLIPTextModel
from model_scope.unet_3d_condition import UNet3DConditionModel

from utils.lora import collapse_lora, monkeypatch_remove_lora
from utils.lora_handler import LoraHandler

import torch
import torchvision
from diffusers.models import AutoencoderKL

from concurrent.futures import ThreadPoolExecutor
import uuid

DESCRIPTION = """# T2V-Turbo 🚀
We provide T2V-Turbo (MS) distilled from [ModelScopeT2V](https://huggingface.co/ali-vilab/text-to-video-ms-1.7b/) with the reward feedback from [HPSv2.1](https://github.com/tgxs002/HPSv2/tree/master) and [ViCLIP](https://huggingface.co/OpenGVLab/ViCLIP).

You can download the the models from [here](https://huggingface.co/jiachenli-ucsb/T2V-Turbo-MS). Check out our [Project page](https://t2v-turbo.github.io) 😄
"""
if torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CUDA 😀</p>"
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    DESCRIPTION += "\n<p>Running on XPU 🤓</p>"
else:
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES") == "1"
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE") == "1"


"""
Operation System Options:
    If you are using MacOS, please set the following (device="mps") ;
    If you are using Linux & Windows with Nvidia GPU, please set the device="cuda";
    If you are using Linux & Windows with Intel Arc GPU, please set the device="xpu";
"""
# device = "mps"    # MacOS
# device = "xpu"    # Intel Arc GPU
device = "cuda"  # Linux & Windows


"""
   DTYPE Options:
      To reduce GPU memory you can set "DTYPE=torch.float16",
      but image quality might be compromised
"""
DTYPE = (
    torch.float16
)  # torch.float16 works as well, but pictures seem to be a bit worse


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def save_video(
    vid_tensor, profile: gr.OAuthProfile | None, metadata: dict, root_path="./"
):
    unique_name = str(uuid.uuid4()) + ".mp4"
    unique_name = os.path.join(root_path, unique_name)

    video = vid_tensor.detach().cpu()
    video = torch.clamp(video.float(), -1.0, 1.0)
    video = video.permute(1, 0, 2, 3)  # t,c,h,w
    video = (video + 1.0) / 2.0
    video = (video * 255).to(torch.uint8).permute(0, 2, 3, 1)

    torchvision.io.write_video(
        unique_name, video, fps=8, video_codec="h264", options={"crf": "10"}
    )
    return unique_name


def save_videos(
    video_array,
    profile: gr.OAuthProfile | None,
    metadata: dict,
):
    paths = []
    root_path = "./videos/"
    os.makedirs(root_path, exist_ok=True)
    with ThreadPoolExecutor() as executor:
        paths = list(
            executor.map(
                save_video,
                video_array,
                [profile] * len(video_array),
                [metadata] * len(video_array),
                [root_path] * len(video_array),
            )
        )
    return paths[0]


def generate(
    prompt: str,
    seed: int = 0,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 4,
    num_frames: int = 16,
    randomize_seed: bool = False,
    param_dtype="torch.float16",
    progress=gr.Progress(track_tqdm=True),
    profile: gr.OAuthProfile | None = None,
):
    seed = randomize_seed_fn(seed, randomize_seed)
    torch.manual_seed(seed)
    pipeline.to(
        torch_device=device,
        torch_dtype=torch.float16 if param_dtype == "torch.float16" else torch.float32,
    )
    start_time = time.time()

    result = pipeline(
        prompt=prompt,
        frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_videos_per_prompt=1,
    )
    paths = save_videos(
        result,
        profile,
        metadata={
            "prompt": prompt,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
        },
    )
    print(time.time() - start_time)
    return paths, seed


examples = [
    "a dog wearing vr goggles on a boat",
    "Darth vader surfing in waves.",
    "Mickey Mouse is dancing on white background",
    "close-up shot, high detailed, A boy with a baseball cap, freckles, and a playful grin.",
    "an old man with a long grey beard and green eyes, camera rotate anticlockwise.",
    "The flowing water sparkled under the golden sunrise in a peaceful mountain river.",
    "close-up shot, high detailed, a girl with long curly blonde hair and sunglasses.",
    "Slow motion steam rises from a hot cup of coffee.",
    "A fluffy teddy bear sits on a bed of soft pillows surrounded by children's toys.",
]


if __name__ == "__main__":
    # Add model name as parameter
    parser = argparse.ArgumentParser(description="Gradio demo for T2V-Turbo.")
    parser.add_argument(
        "--unet_dir",
        type=str,
        help="Directory of the UNet model",
    )
    args = parser.parse_args()

    pretrained_model_path = "ali-vilab/text-to-video-ms-1.7b"
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    teacher_unet = UNet3DConditionModel.from_pretrained(
        pretrained_model_path, subfolder="unet"
    )

    time_cond_proj_dim = 256
    unet = UNet3DConditionModel.from_config(
        teacher_unet.config, time_cond_proj_dim=time_cond_proj_dim
    )
    # load teacher_unet weights into unet
    unet.load_state_dict(teacher_unet.state_dict(), strict=False)
    del teacher_unet
    set_torch_2_attn(unet)
    use_unet_lora = True
    lora_manager = LoraHandler(
        version="cloneofsimo",
        use_unet_lora=use_unet_lora,
        save_for_webui=True,
    )
    lora_manager.add_lora_to_model(
        use_unet_lora,
        unet,
        lora_manager.unet_replace_modules,
        lora_path=args.unet_dir,
        dropout=0.1,
        r=32,
    )
    collapse_lora(unet, lora_manager.unet_replace_modules)
    monkeypatch_remove_lora(unet)
    unet.eval()

    noise_scheduler = T2VTurboScheduler()
    pipeline = T2VTurboMSPipeline(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=noise_scheduler,
    )

    pipeline.to(device)

    with gr.Blocks(css="style.css") as demo:
        gr.Markdown(DESCRIPTION)
        gr.DuplicateButton(
            value="Duplicate Space for private use",
            elem_id="duplicate-button",
            visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
        )
        with gr.Group():
            with gr.Row():
                prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                )
                run_button = gr.Button("Run", scale=0)
            result_video = gr.Video(
                label="Generated Video", interactive=False, autoplay=True
            )
        with gr.Accordion("Advanced options", open=False):
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
                randomize=True,
            )
            randomize_seed = gr.Checkbox(label="Randomize seed across runs", value=True)
            dtype_choices = ["torch.float16", "torch.float32"]
            param_dtype = gr.Radio(
                dtype_choices,
                label="torch.dtype",
                value=dtype_choices[0],
                interactive=True,
                info="To save GPU memory, use torch.float16. For better quality, use torch.float32.",
            )

            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance scale for base",
                    minimum=2,
                    maximum=14,
                    step=0.1,
                    value=7.5,
                )
                num_inference_steps = gr.Slider(
                    label="Number of inference steps for base",
                    minimum=1,
                    maximum=8,
                    step=1,
                    value=4,
                )
            with gr.Row():
                num_frames = gr.Slider(
                    label="Number of Video Frames",
                    minimum=16,
                    maximum=48,
                    step=8,
                    value=16,
                )

        gr.Examples(
            examples=examples,
            inputs=prompt,
            outputs=result_video,
            fn=generate,
            cache_examples=CACHE_EXAMPLES,
        )

        gr.on(
            triggers=[
                prompt.submit,
                run_button.click,
            ],
            fn=generate,
            inputs=[
                prompt,
                seed,
                guidance_scale,
                num_inference_steps,
                num_frames,
                randomize_seed,
                param_dtype,
            ],
            outputs=[result_video, seed],
            api_name="run",
        )

    demo.queue(api_open=False)
    # demo.queue(max_size=20).launch()
    demo.launch()

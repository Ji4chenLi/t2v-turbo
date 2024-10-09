import ast
import gc
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.attention import BasicTransformerBlock
from decord import VideoReader
import wandb


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def is_attn(name):
    return "attn1" or "attn2" == name.split(".")[-1]


def set_processors(attentions):
    for attn in attentions:
        attn.set_processor(AttnProcessor2_0())


def set_torch_2_attn(unet):
    optim_count = 0

    for name, module in unet.named_modules():
        if is_attn(name):
            if isinstance(module, torch.nn.ModuleList):
                for m in module:
                    if isinstance(m, BasicTransformerBlock):
                        set_processors([m.attn1, m.attn2])
                        optim_count += 1
    if optim_count > 0:
        print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")


# From LatentConsistencyModel.get_guidance_scale_embedding
def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
    """
    See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    Args:
        timesteps (`torch.Tensor`):
            generate embedding vectors at these timesteps
        embedding_dim (`int`, *optional*, defaults to 512):
            dimension of the embeddings to generate
        dtype:
            data type of the generated embeddings

    Returns:
        `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


# From LCMScheduler.get_scalings_for_boundary_condition_discrete
def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    scaled_timestep = timestep_scaling * timestep
    c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
    c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5
    return c_skip, c_out


# Compare LCMScheduler.step, Step 4
def get_predicted_original_sample(
    model_output, timesteps, sample, prediction_type, alphas, sigmas
):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_x_0


# Based on step 4 in DDIMScheduler.step
def get_predicted_noise(
    model_output, timesteps, sample, prediction_type, alphas, sigmas
):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_epsilon = model_output
    elif prediction_type == "sample":
        pred_epsilon = (sample - alphas * model_output) / sigmas
    elif prediction_type == "v_prediction":
        pred_epsilon = alphas * model_output + sigmas * sample
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_epsilon


# From LatentConsistencyModel.get_guidance_scale_embedding
def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
    """
    See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    Args:
        timesteps (`torch.Tensor`):
            generate embedding vectors at these timesteps
        embedding_dim (`int`, *optional*, defaults to 512):
            dimension of the embeddings to generate
        dtype:
            data type of the generated embeddings

    Returns:
        `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


# From LCMScheduler.get_scalings_for_boundary_condition_discrete
def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    scaled_timestep = timestep_scaling * timestep
    c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
    c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5
    return c_skip, c_out


# Compare LCMScheduler.step, Step 4
def get_predicted_original_sample(
    model_output, timesteps, sample, prediction_type, alphas, sigmas
):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_x_0


# Based on step 4 in DDIMScheduler.step
def get_predicted_noise(
    model_output, timesteps, sample, prediction_type, alphas, sigmas
):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_epsilon = model_output
    elif prediction_type == "sample":
        pred_epsilon = (sample - alphas * model_output) / sigmas
    elif prediction_type == "v_prediction":
        pred_epsilon = alphas * model_output + sigmas * sample
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_epsilon


def param_optim(model, condition, extra_params=None, is_lora=False, negation=None):
    extra_params = extra_params if len(extra_params.keys()) > 0 else None
    return {
        "model": model,
        "condition": condition,
        "extra_params": extra_params,
        "is_lora": is_lora,
        "negation": negation,
    }


def create_optim_params(name="param", params=None, lr=5e-6, extra_params=None):
    params = {"name": name, "params": params, "lr": lr}
    if extra_params is not None:
        for k, v in extra_params.items():
            params[k] = v

    return params


def create_optimizer_params(model_list, lr):
    import itertools

    optimizer_params = []

    for optim in model_list:
        model, condition, extra_params, is_lora, negation = optim.values()
        # Check if we are doing LoRA training.
        if is_lora and condition and isinstance(model, list):
            params = create_optim_params(
                params=itertools.chain(*model), extra_params=extra_params
            )
            optimizer_params.append(params)
            continue

        if is_lora and condition and not isinstance(model, list):
            for n, p in model.named_parameters():
                if "lora" in n:
                    params = create_optim_params(n, p, lr, extra_params)
                    optimizer_params.append(params)
            continue

        # If this is true, we can train it.
        if condition:
            for n, p in model.named_parameters():
                should_negate = "lora" in n and not is_lora
                if should_negate:
                    continue

                params = create_optim_params(n, p, lr, extra_params)
                optimizer_params.append(params)

    return optimizer_params


def handle_trainable_modules(
    model, trainable_modules=None, is_enabled=True, negation=None
):
    acc = []
    unfrozen_params = 0

    if trainable_modules is not None:
        unlock_all = any([name == "all" for name in trainable_modules])
        if unlock_all:
            model.requires_grad_(True)
            unfrozen_params = len(list(model.parameters()))
        else:
            model.requires_grad_(False)
            for name, param in model.named_parameters():
                for tm in trainable_modules:
                    if all([tm in name, name not in acc, "lora" not in name]):
                        param.requires_grad_(is_enabled)
                        acc.append(name)
                        unfrozen_params += 1


def huber_loss(pred, target, huber_c=0.001):
    loss = torch.sqrt((pred.float() - target.float()) ** 2 + huber_c**2) - huber_c
    return loss.mean()


@torch.no_grad()
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        src_to_dtype = src.to(targ.dtype)
        targ.detach().mul_(rate).add_(src_to_dtype, alpha=1 - rate)


def log_validation_video(pipeline, args, accelerator, save_fps):
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    validation_prompts = [
        "An astronaut riding a horse.",
        "Darth vader surfing in waves.",
        "Robot dancing in times square.",
        "Clown fish swimming through the coral reef.",
        "A child excitedly swings on a rusty swing set, laughter filling the air.",
        "With the style of van gogh, A young couple dances under the moonlight by the lake.",
        "A young woman with glasses is jogging in the park wearing a pink headband.",
        "Impressionist style, a yellow rubber duck floating on the wave on the sunset",
        "Wolf, turns its head, in the wild",
        "Iron man, walks, on the moon, 8k, high detailed, best quality",
    ]

    video_logs = []

    use_motion_cond = getattr(args, "use_motion_cond", False)
    motion_gs = getattr(args, "motion_gs", 0.)
    for _, prompt in enumerate(validation_prompts):
        with torch.autocast("cuda"):
            for i in range(2):
                videos = pipeline(
                    prompt=prompt,
                    frames=args.n_frames,
                    num_inference_steps=8 * (i + 1),
                    num_videos_per_prompt=1,
                    fps=args.fps,
                    use_motion_cond=use_motion_cond,
                    motion_gs=motion_gs,
                    lcm_origin_steps=args.num_ddim_timesteps,
                    generator=generator,
                )
                videos = (videos.clamp(-1.0, 1.0) + 1.0) / 2.0
                videos = (
                    (videos * 255)
                    .to(torch.uint8)
                    .permute(0, 2, 1, 3, 4)
                    .cpu()
                    .numpy()
                )
            video_logs.append(
                {
                    "validation_prompt": f"Steps={4 * (i + 1)}, {prompt}",
                    "videos": videos,
                }
            )

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            formatted_videos = []
            for log in video_logs:
                videos = log["videos"]
                validation_prompt = log["validation_prompt"]
                for video in videos:
                    video = wandb.Video(video, caption=validation_prompt, fps=save_fps)
                    formatted_videos.append(video)

            tracker.log({f"validation": formatted_videos})

        del pipeline
        gc.collect()


def tuple_type(s):
    if isinstance(s, tuple):
        return s
    value = ast.literal_eval(s)
    if isinstance(value, tuple):
        return value
    raise TypeError("Argument must be a tuple")


def load_model_checkpoint(model, ckpt):
    def load_checkpoint(model, ckpt, full_strict):
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
        if "state_dict" in list(state_dict.keys()):
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=full_strict)
        del state_dict
        gc.collect()
        return model

    load_checkpoint(model, ckpt, full_strict=True)
    print(">>> model checkpoint loaded.")
    return model


def read_video_to_tensor(
    path_to_video, sample_fps, sample_frames, uniform_sampling=False
):
    video_reader = VideoReader(path_to_video)
    video_fps = video_reader.get_avg_fps()
    video_frames = video_reader._num_frame
    video_duration = video_frames / video_fps
    sample_duration = sample_frames / sample_fps
    stride = video_fps / sample_fps

    if uniform_sampling or video_duration <= sample_duration:
        index_range = np.linspace(0, video_frames - 1, sample_frames).astype(np.int32)
    else:
        max_start_frame = video_frames - np.ceil(sample_frames * stride).astype(
            np.int32
        )
        if max_start_frame > 0:
            start_frame = random.randint(0, max_start_frame)
        else:
            start_frame = 0

        index_range = start_frame + np.arange(sample_frames) * stride
        index_range = np.round(index_range).astype(np.int32)

    sampled_frames = video_reader.get_batch(index_range).asnumpy()
    pixel_values = torch.from_numpy(sampled_frames).permute(0, 3, 1, 2).contiguous()
    pixel_values = pixel_values / 255.0
    del video_reader

    return pixel_values


def calculate_motion_rank_new(tensor_ref, tensor_gen, rank_k=1):
    if rank_k == 0:
        loss = torch.tensor(0.0, device=tensor_ref.device)
    elif rank_k > tensor_ref.shape[-1]:
        raise ValueError(
            "The value of rank_k cannot be larger than the number of frames"
        )
    else:
        # Sort the reference tensor along the frames dimension
        _, sorted_indices = torch.sort(tensor_ref, dim=-1)
        # Create a mask to select the top rank_k frames
        mask = torch.zeros_like(tensor_ref, dtype=torch.bool)
        mask.scatter_(-1, sorted_indices[..., -rank_k:], True)
        # Compute the mean squared error loss only on the masked elements
        loss = F.mse_loss(tensor_ref[mask].detach(), tensor_gen[mask])
    return loss


def compute_temp_loss(attention_prob, attention_prob_example):
    temp_attn_prob_loss = []
    # 1. Loop though all layers to get the query, key, and Compute the PCA loss
    for name in attention_prob.keys():
        attn_prob_example = attention_prob_example[name]
        attn_prob = attention_prob[name]

        module_attn_loss = calculate_motion_rank_new(
            attn_prob_example.detach(), attn_prob, rank_k=1
        )
        temp_attn_prob_loss.append(module_attn_loss)

    loss_temp = torch.stack(temp_attn_prob_loss) * 100
    loss = loss_temp.mean()
    return loss

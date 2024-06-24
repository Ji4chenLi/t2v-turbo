import numpy as np
import torch

from utils.common_utils import extract_into_tensor


class DDIMSolver:
    def __init__(
        self,
        alpha_cumprods,
        timesteps=1000,
        ddim_timesteps=50,
        scale_a=1.0,
        scale_b=0.7,
        mid_step=400,
        ddim_eta=0.0,
        use_scale=False,
    ):
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (
            np.arange(1, ddim_timesteps + 1) * step_ratio
        ).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

        ## From VideoCrafter 2
        self.use_scale = use_scale
        if use_scale:
            scale_arr1 = np.linspace(scale_a, scale_b, mid_step)
            # VideoCrafter 2 set scale_arr2 in this way, seems to be its bug
            scale_arr2 = np.full(timesteps, scale_b)
            scale_arr = np.concatenate((scale_arr1, scale_arr2))
            self.ddim_scale_arr = scale_arr[self.ddim_timesteps]
            self.ddim_scale_arr_prev = np.asarray(
                [scale_arr[0]] + scale_arr[self.ddim_timesteps[:-1]].tolist()
            )
            self.ddim_scale_arr = torch.from_numpy(self.ddim_scale_arr)
            self.ddim_scale_arr_prev = torch.from_numpy(self.ddim_scale_arr_prev)

            self.ddim_sigmas = ddim_eta * torch.sqrt(
                (1 - self.ddim_alpha_cumprods_prev)
                / (1 - self.ddim_alpha_cumprods)
                * (1 - self.ddim_alpha_cumprods / self.ddim_alpha_cumprods_prev)
            )

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)

        ## From VideoCrafter 2
        if self.use_scale:
            self.ddim_scale_arr = self.ddim_scale_arr.to(device)
            self.ddim_scale_arr_prev = self.ddim_scale_arr_prev.to(device)
            self.ddim_sigmas = self.ddim_sigmas.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(
            self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape
        )
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        if self.use_scale:
            scale_t = extract_into_tensor(
                self.ddim_scale_arr, timestep_index, pred_x0.shape
            )
            scale_t_prev = extract_into_tensor(
                self.ddim_scale_arr_prev, timestep_index, pred_x0.shape
            )
            sigma_t = extract_into_tensor(
                self.ddim_sigmas, timestep_index, pred_x0.shape
            )
            noise = sigma_t * torch.randn_like(pred_x0)
            coef = scale_t_prev / scale_t
            x_prev = alpha_cumprod_prev.sqrt() * coef * pred_x0 + dir_xt + noise
        else:
            x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev


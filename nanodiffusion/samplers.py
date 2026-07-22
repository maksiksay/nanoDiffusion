"""Reverse-process samplers: turn noise back into data.

Part 1 implements the **DDPM ancestral sampler**. Having trained a network to
predict the noise in ``x_t``, we walk backwards from pure noise ``x_T ~ N(0, I)``
to a clean sample ``x_0``, one timestep at a time. Each reverse step:

    1. predict eps = model(x_t, t),
    2. form the posterior mean of x_{t-1} given x_t and that eps,
    3. add a bit of fresh noise (except on the final step to x_0).

    mean       = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps)
    x_{t-1}    = mean + sqrt(posterior_var_t) * z ,   z ~ N(0, I)

where posterior_var_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t).

We record every intermediate x_t so the notebooks can animate the reverse
trajectory.
"""
from __future__ import annotations

import torch

from .schedules import NoiseSchedule


@torch.no_grad()
def ddpm_sample(
    model,
    schedule: NoiseSchedule,
    shape: tuple[int, ...],
    device=None,
    return_trajectory: bool = False,
):
    """Sample from the model by reversing the diffusion process.

    Args:
        model:   trained denoiser predicting eps.
        schedule: the NoiseSchedule used at training time.
        shape:   shape of the batch to generate, e.g. (n, 2) for 2D points.
        device:  where to run; defaults to the model's device.
        return_trajectory: if True, also return a list of x_t at every step
                           (length T+1, from x_T down to x_0), for plotting.

    Returns:
        x_0, or (x_0, trajectory) if return_trajectory.
    """
    if device is None:
        device = next(model.parameters()).device
    schedule = schedule.to(device)
    model.eval()

    x = torch.randn(shape, device=device)
    ndim = x.dim()
    traj = [x.clone()]

    for step in reversed(range(schedule.timesteps)):
        t = torch.full((shape[0],), step, device=device, dtype=torch.long)
        eps = model(x, t)

        beta_t = schedule.gather(schedule.betas, t, ndim)
        alpha_t = schedule.gather(schedule.alphas, t, ndim)
        sqrt_1m_ab = schedule.gather(schedule.sqrt_one_minus_alpha_bars, t, ndim)

        mean = (x - beta_t / sqrt_1m_ab * eps) / torch.sqrt(alpha_t)

        if step > 0:
            ab_t = schedule.gather(schedule.alpha_bars, t, ndim)
            ab_prev = schedule.gather(schedule.alpha_bars_prev, t, ndim)
            posterior_var = beta_t * (1.0 - ab_prev) / (1.0 - ab_t)
            x = mean + torch.sqrt(posterior_var) * torch.randn_like(x)
        else:
            x = mean  # final step: no noise, return the clean estimate

        if return_trajectory:
            traj.append(x.clone())

    return (x, traj) if return_trajectory else x

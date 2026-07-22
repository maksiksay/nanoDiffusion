"""The forward (noising) process q(x_t | x_0).

The whole trick that makes diffusion trainable is the **"nice property"**: thanks
to the reparameterization trick and the fact that a sum of Gaussians is Gaussian,
we can jump directly from a clean sample to its noised version at *any* timestep
``t`` in a single step:

    x_t = sqrt(alpha_bar_t) * x_0  +  sqrt(1 - alpha_bar_t) * eps ,   eps ~ N(0, I)

No need to simulate all t little noising steps. This is what lets us train on a
random ``t`` each iteration.
"""
from __future__ import annotations

import torch

from .schedules import NoiseSchedule


def add_noise(
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Corrupt ``x0`` to timestep ``t`` in one shot.

    Args:
        x0:        clean data, shape (B, ...).
        t:         (B,) LongTensor of timesteps in [0, T-1].
        schedule:  the NoiseSchedule providing alpha_bar_t.
        noise:     optional pre-sampled eps (same shape as x0). If omitted, drawn
                   from N(0, I). Returned so the training loss can use the exact
                   eps the network is asked to predict.

    Returns:
        (x_t, noise)
    """
    if noise is None:
        noise = torch.randn_like(x0)
    ndim = x0.dim()
    sqrt_ab = schedule.gather(schedule.sqrt_alpha_bars, t, ndim)
    sqrt_1m_ab = schedule.gather(schedule.sqrt_one_minus_alpha_bars, t, ndim)
    x_t = sqrt_ab * x0 + sqrt_1m_ab * noise
    return x_t, noise

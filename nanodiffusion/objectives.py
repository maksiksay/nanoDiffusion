"""Training objectives.

Part 1 uses the DDPM **epsilon-prediction** loss, which is startlingly simple:

    1. take a clean batch x_0,
    2. pick a random timestep t for each sample,
    3. noise it to x_t with a known eps (the "nice property"),
    4. ask the network to predict eps from (x_t, t),
    5. minimise the mean-squared error between predicted and true eps.

That's it. The deep result (from the DDPM paper) is that this simple regression
objective is a re-weighting of the true variational bound on the data likelihood
-- so minimising MSE on noise really does train a generative model.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .forward import add_noise
from .schedules import NoiseSchedule


def sample_timesteps(batch_size: int, schedule: NoiseSchedule, device) -> torch.Tensor:
    """Uniformly sample one timestep in [0, T-1] per item in the batch."""
    return torch.randint(0, schedule.timesteps, (batch_size,), device=device)


def ddpm_eps_loss(model, x0: torch.Tensor, schedule: NoiseSchedule) -> torch.Tensor:
    """The DDPM epsilon-prediction MSE loss for one batch of clean data ``x0``."""
    t = sample_timesteps(x0.shape[0], schedule, x0.device)
    x_t, noise = add_noise(x0, t, schedule)
    predicted = model(x_t, t)
    return F.mse_loss(predicted, noise)

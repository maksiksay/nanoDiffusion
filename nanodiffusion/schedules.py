"""Noise (variance) schedules.

A diffusion model corrupts data over ``T`` discrete steps. At each step ``t`` we
add a little Gaussian noise with variance ``beta_t``. Everything else we need is
derived from the betas:

    alpha_t      = 1 - beta_t
    alpha_bar_t  = prod_{s<=t} alpha_s     ("how much of the signal survives to t")

``alpha_bar_t`` is the key quantity: it lets us jump straight from a clean sample
``x_0`` to a noised sample ``x_t`` in one shot (the "nice property", see
``forward.py``), instead of simulating all t steps.

``NoiseSchedule`` precomputes these arrays once and hands out per-timestep values
with the right broadcasting shape for whatever tensor rank your data has.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def linear_beta_schedule(timesteps: int, beta_start=1e-4, beta_end=0.02) -> torch.Tensor:
    """The original DDPM schedule: betas increase linearly with t."""
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """The 'improved DDPM' cosine schedule (Nichol & Dhariwal, 2021).

    Signal is destroyed more gently early on, which tends to help sample quality.
    We define alpha_bar directly from a cosine, then read the betas back out.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alpha_bars = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alpha_bars = alpha_bars / alpha_bars[0]
    betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
    return torch.clamp(betas, 1e-8, 0.999)


@dataclass
class NoiseSchedule:
    """Holds the betas and everything derived from them."""

    betas: torch.Tensor  # (T,)

    def __post_init__(self):
        self.timesteps = int(self.betas.shape[0])
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        # alpha_bar of the *previous* step, with alpha_bar_{-1} := 1.0
        self.alpha_bars_prev = torch.cat(
            [torch.ones(1, dtype=self.betas.dtype), self.alpha_bars[:-1]]
        )
        # Handy precomputed roots used by the forward process and samplers.
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    @classmethod
    def make(cls, kind: str = "cosine", timesteps: int = 200) -> "NoiseSchedule":
        if kind == "linear":
            betas = linear_beta_schedule(timesteps)
        elif kind == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown schedule kind {kind!r}")
        return cls(betas)

    def to(self, device) -> "NoiseSchedule":
        """Move all cached tensors to ``device`` (returns self for chaining)."""
        for name, val in list(self.__dict__.items()):
            if isinstance(val, torch.Tensor):
                setattr(self, name, val.to(device))
        return self

    def gather(self, tensor: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        """Pick ``tensor[t]`` and reshape to broadcast against an ``ndim``-D batch.

        ``t`` is a (B,) LongTensor of timesteps; the result is (B, 1, ..., 1) so it
        multiplies cleanly with data of shape (B, ...).
        """
        out = tensor.to(t.device)[t]
        return out.reshape(t.shape[0], *([1] * (ndim - 1)))

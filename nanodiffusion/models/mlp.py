"""A tiny MLP denoiser for 2D data.

Given a noised point ``x_t`` (shape (B, 2)) and its timestep ``t``, predict the
noise ``eps`` that was added. This is the smallest network that makes diffusion
work end-to-end, so it is where we learn the training loop and the sampler before
any convolutions enter the picture.

The timestep is embedded (see ``blocks.SinusoidalTimeEmbedding``), passed through
a small MLP, and concatenated into the network at the input so every layer knows
the noise level.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import SinusoidalTimeEmbedding


class MLPDenoiser(nn.Module):
    def __init__(
        self,
        data_dim: int = 2,
        hidden: int = 128,
        depth: int = 4,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )
        layers = [nn.Linear(data_dim + time_embed_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, data_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict eps for noised inputs ``x`` at timesteps ``t``."""
        temb = self.time_mlp(t)
        return self.net(torch.cat([x, temb], dim=-1))

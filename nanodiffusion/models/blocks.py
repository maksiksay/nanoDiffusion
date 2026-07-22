"""Reusable network building blocks.

For Part 1 we only need one: a way to turn the integer timestep ``t`` into a
vector the network can consume. A denoiser must behave *differently* at different
noise levels, so ``t`` is a first-class input. We embed it the same way
transformers embed positions — with sinusoids of many frequencies — which gives
the network a smooth, expressive encoding of "how noisy is this?".
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Map a (B,) tensor of timesteps to a (B, dim) sinusoidal embedding."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dim must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        # frequencies geometrically spaced from 1 down to ~1e-4
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

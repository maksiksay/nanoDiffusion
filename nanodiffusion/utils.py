"""Small, dependency-light helpers used everywhere: device, seeding, plotting."""
from __future__ import annotations

import random

import numpy as np
import torch


def pick_device() -> torch.device:
    """Return the best available device: CUDA, then Apple MPS, then CPU.

    The whole library is written to run on whatever this returns, so the same
    code trains on an NVIDIA GPU, an Apple-Silicon Mac, or a laptop CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy and PyTorch so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scatter_2d(ax, points, title: str = "", lim: float = 2.5, **kw):
    """Scatter a set of 2D points onto a matplotlib axis, with sane defaults.

    ``points`` may be a torch.Tensor or a NumPy array of shape (N, 2).
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    defaults = dict(s=4, alpha=0.5)
    defaults.update(kw)
    ax.scatter(points[:, 0], points[:, 1], **defaults)
    ax.set_title(title)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    return ax

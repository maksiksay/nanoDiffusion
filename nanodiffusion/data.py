"""Toy 2D datasets.

These are the "hello world" of diffusion: distributions in the plane that you can
plot in their entirety. Because you can *see* the whole data distribution, the
2D toys double as a regression test for the full train -> sample pipeline before
any GPU-hours are spent on images.

Every generator returns a float32 tensor of shape (n, 2), roughly zero-mean and
unit-scale, so the same noise schedule works across datasets.
"""
from __future__ import annotations

import numpy as np
import torch


def _normalize(x: np.ndarray) -> torch.Tensor:
    x = x - x.mean(axis=0, keepdims=True)
    x = x / (x.std() + 1e-8)
    return torch.from_numpy(x.astype(np.float32))


def swiss_roll(n: int) -> torch.Tensor:
    """A 2D swiss-roll spiral."""
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n))
    x = t * np.cos(t)
    y = t * np.sin(t)
    pts = np.stack([x, y], axis=1) + 0.3 * np.random.randn(n, 2)
    return _normalize(pts)


def spirals(n: int) -> torch.Tensor:
    """Two interleaved spirals (a classic 'hard to separate' shape)."""
    n_per = n // 2
    t = np.sqrt(np.random.rand(n_per)) * 3.0 * np.pi
    a = np.stack([t * np.cos(t), t * np.sin(t)], axis=1)
    b = np.stack([t * np.cos(t + np.pi), t * np.sin(t + np.pi)], axis=1)
    pts = np.concatenate([a, b], axis=0) + 0.2 * np.random.randn(2 * n_per, 2)
    return _normalize(pts)


def moons(n: int) -> torch.Tensor:
    """Two half-moons."""
    n_per = n // 2
    t = np.pi * np.random.rand(n_per)
    outer = np.stack([np.cos(t), np.sin(t)], axis=1)
    inner = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], axis=1)
    pts = np.concatenate([outer, inner], axis=0) + 0.06 * np.random.randn(2 * n_per, 2)
    return _normalize(pts)


_REGISTRY = {"swiss_roll": swiss_roll, "spirals": spirals, "moons": moons}


def toy2d(name: str = "swiss_roll", n: int = 8000) -> torch.Tensor:
    """Return ``n`` samples from the named 2D toy distribution."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown toy dataset {name!r}; choose from {list(_REGISTRY)}")
    return _REGISTRY[name](n)

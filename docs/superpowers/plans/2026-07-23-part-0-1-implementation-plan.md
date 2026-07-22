# nanoDiffusion — Part 0–1 Implementation Plan

**Date:** 2026-07-23
**Design doc:** [../specs/2026-07-23-nanodiffusion-design.md](../specs/2026-07-23-nanodiffusion-design.md)
**Scope:** Package skeleton + DDPM on 2D toy data (notebooks 00–03).

## What "done" means for Part 0–1

You can open the notebooks, run every cell on your M3 Pro (CPU/MPS), and watch a
2D swiss roll dissolve into noise and be reconstructed by a diffusion model you
trained. The `nanodiffusion` package contains the graduated, reusable code. This
is the whole-pipeline regression test the rest of the project builds on.

## Environment

- `uv` virtual env pinned to **Python 3.12** (solid PyTorch/MPS wheels).
- Deps: `torch`, `numpy`, `matplotlib`, `pyyaml`, `tqdm`, `jupyter`, `ipykernel`.
- Device auto-detect: `cuda → mps → cpu` (M3 Pro uses MPS).

## Step 1 — Environment & scaffolding

- `uv venv --python 3.12`; install deps; register a Jupyter kernel.
- Directory layout per design doc (`nanodiffusion/`, `notebooks/`, `configs/`,
  `docs/`).
- `.gitignore` (venv, `__pycache__`, data, checkpoints, `.ipynb_checkpoints`),
  `pyproject.toml`, `README.md`, `configs/tiny.yaml`.

## Step 2 — Package modules needed for Part 1

Each is small and single-purpose. Built to be readable, then imported by the
notebooks (the notebooks teach them inline first, then rely on the package).

- `utils.py` — `pick_device()`, `set_seed()`, plotting helpers for 2D data.
- `data.py` — `toy2d(name, n)` returning swiss roll / spirals / moons as a
  normalized `(N, 2)` tensor.
- `schedules.py` — `linear_beta_schedule`, `cosine_beta_schedule`, and a
  `NoiseSchedule` holding `betas, alphas, alpha_bars` + broadcast helpers.
- `forward.py` — `add_noise(x0, t, schedule, noise)` implementing
  `q(x_t | x_0)` via the reparameterization trick (the "nice property").
- `models/blocks.py` — `SinusoidalTimeEmbedding`.
- `models/mlp.py` — `MLPDenoiser(x, t)` → predicted ε for 2D points.
- `objectives.py` — `ddpm_eps_loss(model, x0, schedule)` (sample t, noise,
  predict, MSE).
- `samplers.py` — `ddpm_sample(model, schedule, shape)` ancestral sampler,
  returning the full trajectory for plotting.

## Step 3 — Notebooks

- `00_intuition.ipynb` — the idea in pictures: forward corruption of a swiss roll,
  the reverse goal. Markdown-heavy, minimal code (uses `data.toy2d`).
- `01_forward_process.ipynb` — derive/plot schedules; reparameterization trick;
  show `x_t` at increasing `t`; sanity asserts (monotonic ᾱ, `x_T` ≈ unit
  variance). Graduates `schedules.py` + `forward.py`.
- `02_reverse_and_loss.ipynb` — what the net predicts (ε), derive the MSE loss,
  build `MLPDenoiser` inline, train it, plot loss going down; overfit-one-batch
  check. Graduates `models/` + `objectives.py`.
- `03_sampling.ipynb` — derive the ancestral sampler, animate 2D noise → swiss
  roll, compare to the true distribution. Graduates `samplers.py`.

## Step 4 — Verification (the regression test)

A small script/cell trains for a short run on swiss-roll and samples; success =
generated points visibly match the swiss-roll manifold. Run it end-to-end before
declaring Part 0–1 complete. Include the cheap `assert`s from the design doc.

## Out of scope for this part

U-Net, MNIST/CIFAR, DDIM, guidance, score/SDE, flow matching — those are Parts
2–4, later plans.

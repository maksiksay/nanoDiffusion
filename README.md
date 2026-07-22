# nanoDiffusion

A minimal, from-scratch, **educational** implementation of diffusion models — in
the spirit of Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT). Learn how
diffusion models work by building one, step by step, on data you can *see*.

The journey follows the field's own history, from **DDPM (2020)** to modern
**flow matching (2023+)**, on progressively harder data:
**2D toy data → MNIST → CIFAR-10**, reusing the same clean code.

## The path

Read the [design doc](docs/superpowers/specs/2026-07-23-nanodiffusion-design.md)
for the full arc. The notebooks build it up concept by concept:

| Notebook | Concept |
|----------|---------|
| `00_intuition` | What is a diffusion model? (pictures) |
| `01_forward_process` | Noising, the reparameterization trick, schedules |
| `02_reverse_and_loss` | ε-prediction and the simple MSE loss |
| `03_sampling` | The DDPM sampler — noise → data |
| `04_unet_mnist` | A minimal U-Net; first real images |
| `05_ddim_sampling` | Fast, deterministic sampling |
| `06_guidance` | Classifier-free guidance (conditional generation) |
| `07_score_and_sde` | The score / SDE unifying view |
| `08_flow_matching` | The modern formulation |
| `09_cifar10` | The north star |
| `10_custom_dataset` | Your own images |

**Status:** Part 0–1 (notebooks 00–03, DDPM on 2D toy data) — in progress.

## Setup

```bash
uv venv --python 3.12
uv pip install -e ".[notebooks]"
uv run python -m ipykernel install --user --name nanodiffusion
uv run jupyter lab   # open notebooks/
```

The code auto-detects your device (`cuda → mps → cpu`), so it runs on an Apple
Silicon Mac (MPS), an NVIDIA GPU, or Colab with no changes.

## The package

`nanodiffusion/` holds the reusable machinery the notebooks build up — one
concept per file. See the design doc for the module map.

# nanoDiffusion — Design Doc

**Date:** 2026-07-23
**Status:** Approved (design phase complete)

## Goal

An educational, from-scratch implementation of diffusion models in the spirit of
Karpathy's nanoGPT. The learner (solid ML background, new to diffusion) wants to
**understand diffusion deeply, step by step**, and build a **real working
CIFAR-10 image generator**, culminating in a custom toy image dataset. The full
theoretical arc from 2015 → 2025 is in scope, ending with modern
**flow matching**.

Two deliverables emerge together:
1. A sequence of **annotated Jupyter notebooks** that teach each concept with
   math (markdown), plots, and fast-running cells.
2. A clean, minimal **nanoDiffusion Python package** — the polished,
   nanoGPT-style artifact — that the notebooks build up incrementally.

## Approach

**Concept-first "spiral," dataset-staged (Approach A).** Build the core
diffusion machinery once and re-run the *same code* on progressively harder data
by swapping a config: **2D toy data → MNIST → CIFAR-10**. Each new theory lens
(DDIM, guidance, score/SDE, flow matching) is a new notebook that imports the
shared modules. The notebook *order* follows history, but the plumbing is written
once. Concepts are learned on CPU/MPS-speed toy and MNIST data; only the final
CIFAR-10 run needs a real GPU.

## Curriculum (notebook sequence)

Each notebook is self-contained, builds one concept, runs fast on toy/MNIST data,
and imports shared modules.

**Part 0 — Foundations**
- `00_intuition.ipynb` — What a diffusion model is: destroy data with noise, learn
  to reverse it. Visual-only, sets up the arc.

**Part 1 — DDPM on 2D toy data** (everything plottable)
- `01_forward_process.ipynb` — Forward noising, reparameterization trick, the
  "nice property" (jump to any `t` in one step), variance schedules
  (linear/cosine). Plot a swiss roll dissolving into Gaussian noise.
- `02_reverse_and_loss.ipynb` — What the network predicts (ε-prediction), deriving
  the simple MSE loss, why it works. Train a tiny MLP to denoise 2D points.
- `03_sampling.ipynb` — DDPM ancestral sampler. Watch 2D noise become a swiss roll.

**Part 2 — Scaling to images**
- `04_unet_mnist.ipynb` — Build a minimal U-Net from scratch (time embeddings,
  residual blocks, attention). Train on MNIST. First real generated images.
- `05_ddim_sampling.ipynb` — DDIM: deterministic, 10–50× faster sampling. Same
  trained model, better sampler.
- `06_guidance.ipynb` — Classifier-free guidance for conditional generation
  (generate a specific digit).

**Part 3 — The unifying theory**
- `07_score_and_sde.ipynb` — The score-based / SDE view (Song et al.): diffusion
  as reversing an SDE; the reveal that DDPM is secretly estimating the score
  ∇ₓ log p(x). Placed here (after DDPM) so the unification lands with intuition
  already in hand. May be split into 07a (denoising score matching + Langevin)
  and 07b (the SDE view) if it wants more space.

**Part 4 — Modern formulation & the real target**
- `08_flow_matching.ipynb` — Flow matching / rectified flow (SD3-era): generation
  as learning a velocity field. Reuses the shared modules; short because only the
  objective + sampler swap.
- `09_cifar10.ipynb` — Train the polished model on CIFAR-10 (the north star).
  Config-driven, checkpointing, Colab-friendly, resumable.
- `10_custom_dataset.ipynb` — Point the pipeline at a custom toy image set.

**Byproduct:** by notebook 09 the shared modules *are* the clean nanoDiffusion
repo — a nanoGPT-style artifact the learner built themselves.

## Code Architecture

Notebooks stay short because the machinery lives in a small, clean importable
package. Notebooks define a class inline first (to teach it), then "graduate" it
into the package — so each module is seen born before it is imported.

```
nanodiffusion/
├── nanodiffusion/            # the importable package
│   ├── schedules.py          # noise schedules: linear, cosine; ᾱ_t helpers
│   ├── forward.py            # q(x_t | x_0): add_noise(x0, t) via reparam trick
│   ├── models/
│   │   ├── mlp.py            # tiny MLP denoiser (for 2D toy data)
│   │   ├── unet.py          # minimal U-Net: time-embed, resblocks, attention
│   │   └── blocks.py        # SinusoidalTimeEmbedding, ResBlock, Attn
│   ├── objectives.py        # loss fns: eps-pred (DDPM), v-pred, flow-matching
│   ├── samplers.py          # DDPM ancestral, DDIM, ODE/SDE, flow-matching Euler
│   ├── guidance.py          # classifier-free guidance wrapper
│   ├── data.py              # toy2d (swiss roll/spirals), MNIST, CIFAR-10, custom
│   └── utils.py             # seeding, EMA, checkpoints, image grids, device pick
├── notebooks/               # 00..10, the learning path
├── configs/                 # tiny.yaml, mnist.yaml, cifar10.yaml
├── train.py                 # nanoGPT-style CLI: `python train.py --config cifar10`
├── sample.py                # generate from a checkpoint
└── README.md
```

**Design principles:**
- **One concept per file.** `forward.py` only adds noise; `samplers.py` only
  removes it. Any file is readable in one sitting.
- **The objective is swappable.** ε-prediction, v-prediction, and flow-matching
  share the same U-Net and training loop — only `objectives.py` + `samplers.py`
  change. This is *why* notebook 08 is short, and the pedagogical point made
  concrete.
- **Config-driven scale.** Same code, `tiny → mnist → cifar10` by config, like
  nanoGPT.
- **Device-agnostic.** Auto-detect `cuda → mps → cpu`, so identical code runs on
  the M3 Pro (MPS) and on Colab (CUDA) with no changes.

## Tech Stack

Deliberately minimal — nanoGPT ethos, no heavy frameworks:
- **PyTorch** — the only real dependency; U-Net, schedules, samplers all
  hand-written. No `diffusers`, Lightning, or HF training loops.
- **matplotlib** — plotting forward/reverse process, loss curves, sample grids.
- **numpy** — light array work in the 2D notebooks.
- **tqdm** — progress bars.
- **PyYAML** — configs.
- *(optional, later)* **torch-fidelity or a tiny FID impl** — only if quantitative
  evaluation is added.

## Compute Plan

Learner has an **M3 Pro MacBook** (MPS) for local work and will use
**Colab / cloud GPU** for CIFAR-10.

| Stage | Data | Hardware | Time | Purpose |
|-------|------|----------|------|---------|
| 2D toy (nb 01–03, 07, 08) | swiss roll | M3 Pro (CPU/MPS) | seconds–1 min/run | learn concepts, instant cells |
| MNIST (nb 04–06) | 28×28 gray | M3 Pro (MPS) | ~5–15 min | first real images |
| CIFAR-10 (nb 09) | 32×32 color | Colab / cloud GPU | a few hours | the north star |

~90% of the learning happens at CPU/MPS speed on toy/MNIST data; CIFAR-10 is the
last mile where GPU matters. nb 09 is Colab-friendly (checkpoint to Drive,
resumable).

## Validation (how we know it works)

Pedagogical correctness checks — small, visual, confidence-building; no pytest
suite or CI (that would be ceremony against the learning goal).

- **Sanity assertions in modules.** `add_noise` at `t=T` ≈ unit-variance noise;
  ᾱ schedules monotonic and end near 0. Cheap `assert`s that catch classic bugs
  (off-by-one on `t`, wrong broadcasting of `ᾱ_t`).
- **Visual validation is primary.** Forward process should visibly dissolve data;
  reverse should visibly reconstruct; loss should drop. Each notebook ends with a
  "did it work?" plot.
- **2D toy set as a whole-pipeline regression test.** If train→sample recovers the
  swiss roll, the machinery is correct — verified *before* spending GPU hours.
- **Overfit-one-batch check.** Confirm the model can memorize a tiny batch before
  a real run.
- **Optional quantitative eval (later).** A small FID for CIFAR-10 if a number is
  wanted; not required to "finish."

## Out of Scope (YAGNI)

- Production serving, web UI, or API.
- Latent diffusion / VAE encoders (start in pixel space; a possible future
  extension, not part of this arc).
- Text-to-image / large text encoders.
- Distributed / multi-GPU training.
- A formal pytest suite or CI pipeline.

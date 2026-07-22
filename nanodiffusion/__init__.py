"""nanoDiffusion — a minimal, educational diffusion-model library.

One concept per module:

- ``utils``      : device selection, seeding, small plotting helpers.
- ``data``       : toy 2D distributions (and later, image datasets).
- ``schedules``  : noise (variance) schedules and their derived quantities.
- ``forward``    : the forward noising process q(x_t | x_0).
- ``models``     : denoiser networks (MLP now; U-Net later).
- ``objectives`` : training losses (DDPM epsilon-prediction now).
- ``samplers``   : reverse-process samplers (DDPM ancestral now).

The notebooks build these up piece by piece; this package is where the
"graduated" code lives.
"""

__version__ = "0.1.0"

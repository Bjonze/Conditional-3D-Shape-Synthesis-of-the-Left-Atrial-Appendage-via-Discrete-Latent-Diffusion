"""Conditional 3D shape synthesis of the left atrial appendage.

Two stages:

``laa_ldm.codec``
    A conditional 3D VQ-GAN that compresses a 128^3 binary LAA mask into an
    8x8x8 grid of discrete codes.

``laa_ldm.diffusion``
    A discrete (mask-and-replace) latent diffusion model over those codes,
    conditioned on 18 anatomical shape descriptors.
"""

__version__ = "1.0.0"

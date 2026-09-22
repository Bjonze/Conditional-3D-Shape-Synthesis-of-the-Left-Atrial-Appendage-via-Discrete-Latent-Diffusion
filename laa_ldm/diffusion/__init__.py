"""Stage 2: descriptor-conditioned discrete diffusion over the VQ-GAN codes."""

from laa_ldm.diffusion.diffusion import DiffusionTransformer
from laa_ldm.diffusion.model import ConditionalLatentDiffusion

__all__ = ["ConditionalLatentDiffusion", "DiffusionTransformer"]

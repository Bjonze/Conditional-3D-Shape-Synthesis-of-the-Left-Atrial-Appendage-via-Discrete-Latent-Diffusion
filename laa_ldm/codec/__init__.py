"""Stage 1: the conditional 3D VQ-GAN that discretises LAA masks."""

from laa_ldm.codec.inference import VQGANInference, load_vqgan
from laa_ldm.codec.model import VQGAN3D
from laa_ldm.codec.quantize import EMAVectorQuantizer3D

__all__ = ["VQGAN3D", "VQGANInference", "load_vqgan", "EMAVectorQuantizer3D"]

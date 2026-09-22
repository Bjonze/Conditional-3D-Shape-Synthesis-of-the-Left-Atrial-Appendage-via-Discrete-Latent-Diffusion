"""The bridge between the two stages: decoding latent codes back to masks.

During diffusion training the content stream is already discrete - the token
grids are exported once with ``scripts/encode_dataset.py`` - so the codec here
only has to turn sampled index grids back into occupancy volumes.  The frozen
VQ-GAN decoder does that.
"""

import torch
import torch.nn as nn
from einops import rearrange

from laa_ldm.codec.inference import load_vqgan

__all__ = ["BaseCodec", "VQGAN3DCodec"]


class BaseCodec(nn.Module):
    """Interface shared by the content and condition codecs."""

    def get_tokens(self, x, **kwargs):
        """Encode ``x`` into ``{'token': (B, L), ...}``."""
        raise NotImplementedError

    def decode(self, tokens):
        raise NotImplementedError

    def forward(self, **kwargs):
        raise NotImplementedError

    def train(self, mode=True):
        self.training = mode
        return super().train(mode and self.trainable)

    def _set_trainable(self):
        if not self.trainable:
            for _, p in self.named_parameters():
                p.requires_grad = False
            self.eval()


class LatentDecoder(nn.Module):
    """Codebook indices ``(B, D*H*W)`` -> occupancy probabilities ``(B, 1, D, H, W)``."""

    def __init__(self, decoder, post_quant_conv, quantize, d=8, h=8, w=8):
        super().__init__()
        self.decoder = decoder
        self.post_quant_conv = post_quant_conv
        self.quantize = quantize
        self.d, self.h, self.w = d, h, w

    def forward(self, indices):
        b, _ = indices.shape
        z_q = self.quantize.embedding.weight[indices]                 # (B, L, C)
        z_q = z_q.view(b, self.d, self.h, self.w, -1)
        z_q = rearrange(z_q, 'b d h w c -> b c d h w').contiguous()
        return torch.sigmoid(self.decoder(self.post_quant_conv(z_q)))


class VQGAN3DCodec(BaseCodec):
    """Frozen stage-1 VQ-GAN used as the content codec of the diffusion model.

    Args:
        config_path: the VQ-GAN training config (``configs/vqgan3d.yaml``).
        ckpt_path: a VQ-GAN checkpoint.
        token_shape: latent grid, ``(depth, height, width)``.
        num_tokens: codebook size.
        trainable: keep this False - the codec is frozen during stage 2.
    """

    def __init__(
        self,
        config_path,
        ckpt_path,
        token_shape=(8, 8, 8),
        num_tokens=512,
        trainable=False,
    ):
        super().__init__()
        vqgan = load_vqgan(config_path, ckpt_path)

        self.dec = LatentDecoder(vqgan.decoder, vqgan.post_quant_conv, vqgan.quantize,
                                 *token_shape)
        self.quantize = vqgan.quantize

        self.num_tokens = num_tokens
        self.token_shape = list(token_shape)
        self.trainable = trainable
        self._set_trainable()

    @property
    def device(self):
        return self.dec.post_quant_conv.weight.device

    def get_tokens(self, tokens, **kwargs):
        """Identity: the dataset already stores pre-computed code indices.

        See ``scripts/encode_dataset.py``, which runs the conditioned encoder
        over the mask dataset once and writes the index grids to disk.
        """
        return {'token': tokens}

    def decode(self, img_seq):
        """Codebook indices ``(B, L)`` -> occupancy probabilities ``(B, 1, D, H, W)``."""
        return self.dec(img_seq)

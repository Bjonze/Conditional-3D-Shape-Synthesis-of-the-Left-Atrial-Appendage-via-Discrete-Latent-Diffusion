"""Conditioning codec for the descriptor vector.

The diffusion model consumes conditioning through a codec, mirroring how the
content stream goes through the VQ-GAN.  Shape descriptors are already
continuous vectors, so this codec only validates and passes them through; the
learned part lives in :class:`laa_ldm.diffusion.embeddings.ScalarTokenizer`,
which the denoising transformer owns.
"""

import torch

from laa_ldm.diffusion.codec import BaseCodec

__all__ = ["DescriptorCodec"]


class DescriptorCodec(BaseCodec):
    """Pass a descriptor vector through as conditioning tokens.

    Args:
        context_length: number of descriptors expected per sample (18).
        with_mask: also return an all-ones attention mask, kept for symmetry
            with the content codec interface.
    """

    def __init__(self, context_length: int = 18, with_mask: bool = True):
        super().__init__()
        self.context_length = context_length
        self.with_mask = with_mask
        self.trainable = False
        self._set_trainable()

    def __repr__(self):
        return "DescriptorCodec(context_length={}, with_mask={})".format(
            self.context_length, self.with_mask)

    def get_tokens(self, descriptors, **kwargs):
        """Descriptors ``(B, F)`` or ``(B, F, 1)`` -> ``{'token': ..., 'mask': ...}``."""
        if not torch.is_tensor(descriptors):
            descriptors = torch.as_tensor(descriptors, dtype=torch.float32)
        descriptors = descriptors.float()
        if descriptors.dim() == 2:
            descriptors = descriptors.unsqueeze(-1)
        if descriptors.shape[1] != self.context_length:
            raise ValueError("Expected {} descriptors per sample, got {}.".format(
                self.context_length, descriptors.shape[1]))

        tokens = {"token": descriptors}
        if self.with_mask:
            tokens["mask"] = torch.ones(descriptors.shape[0], self.context_length,
                                        dtype=torch.bool, device=descriptors.device)
        return tokens

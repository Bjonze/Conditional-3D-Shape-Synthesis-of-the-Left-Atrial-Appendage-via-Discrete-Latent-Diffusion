"""Loading a trained VQ-GAN for inference.

The training module (:class:`laa_ldm.codec.model.VQGAN3D`) carries the
discriminator and the perceptual loss, which are useless once training is done.
:class:`VQGANInference` holds only the parts needed to encode and decode, and
uses the same submodule names, so a training checkpoint loads straight into it.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from monai.utils import ensure_tuple_rep

from laa_ldm.codec.conditioning import ScalarTokenizer
from laa_ldm.codec.nets import Decoder, Encoder
from laa_ldm.codec.quantize import EMAVectorQuantizer3D
from laa_ldm.utils.config import load_yaml_config

__all__ = ["VQGANInference", "load_vqgan"]


class VQGANInference(nn.Module):
    """Encoder / quantiser / decoder of a trained VQ-GAN, without the training heads.

    Accepts the same ``model:`` block as the VQ-GAN training config.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int = 1,
        out_channels: int = 1,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        num_channels: Sequence[int] = (32, 64, 128, 256),
        attention_levels: Sequence[bool] = (False, False, False, True),
        codebook_dim: int = 256,
        codebook_size: int = 512,
        context_dim: int | None = None,
        num_context_vars: int | None = None,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        with_encoder_nonlocal_attn: bool = True,
        with_decoder_nonlocal_attn: bool = True,
        use_flash_attention: bool = False,
        use_convtranspose: bool = False,
        beta: float = 0.25,
        ema_decay: float = 0.99,
        distance_metric: str = "cosine",
        **unused,
    ) -> None:
        super().__init__()
        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(num_channels))

        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.encoder = Encoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_channels=num_channels,
            out_channels=codebook_dim,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_encoder_nonlocal_attn,
            use_flash_attention=use_flash_attention,
            context_dim=context_dim,
        )
        self.decoder = Decoder(
            spatial_dims=spatial_dims,
            num_channels=num_channels,
            in_channels=codebook_dim,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_decoder_nonlocal_attn,
            use_flash_attention=use_flash_attention,
            use_convtranspose=use_convtranspose,
        )
        self.quant_conv = nn.Conv3d(codebook_dim, codebook_dim, kernel_size=1, stride=1, padding=0)
        self.post_quant_conv = nn.Conv3d(codebook_dim, codebook_dim, kernel_size=1, stride=1,
                                         padding=0)
        self.quantize = EMAVectorQuantizer3D(codebook_size, codebook_dim, beta, decay=ema_decay,
                                             distance_metric=distance_metric)
        self.context_projector = ScalarTokenizer(
            num_features=num_context_vars,
            token_dim=context_dim,
            hidden=32,
            use_transformer=True,
            n_heads=4,
            n_layers=1,
            ff_mult=2,
            dropout=0.0,
        )

    @torch.no_grad()
    def encode(self, x, cond):
        """Mask (B, 1, D, H, W) + descriptors (B, 18, 1) -> quantised latent, indices."""
        ctx = self.context_projector(cond)
        h = self.quant_conv(self.encoder(x, ctx))
        return self.quantize(h)

    @torch.no_grad()
    def decode(self, quant):
        """Quantised latent -> mask logits."""
        return self.decoder(self.post_quant_conv(quant))


def load_vqgan(config_path, ckpt_path, map_location="cpu"):
    """Instantiate :class:`VQGANInference` from a training config and checkpoint.

    Weights are loaded non-strictly: the checkpoint also holds the
    discriminator and perceptual-loss parameters, which have no counterpart here.
    """
    config = load_yaml_config(config_path)
    model = VQGANInference(**config["model"])

    checkpoint = torch.load(ckpt_path, map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, _ = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Checkpoint {ckpt_path} is missing codec weights: {missing[:8]}")
    model.eval()
    return model

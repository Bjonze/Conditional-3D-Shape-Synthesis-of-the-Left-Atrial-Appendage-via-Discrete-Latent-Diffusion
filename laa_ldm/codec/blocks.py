"""Convolutional building blocks for the 3D VQ-GAN.

Derived from the MONAI GenerativeModels ``AutoencoderKL`` blocks (Apache-2.0),
extended with FiLM conditioning on the 18 shape descriptors and a gated skip
connection in the decoder.
"""

from __future__ import annotations

import importlib.util
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from monai.networks.blocks import Convolution

# Optional memory-efficient attention. Enabled with ``use_flash_attention: true``.
if importlib.util.find_spec("xformers") is not None:
    import xformers
    import xformers.ops

    has_xformers = True
else:
    xformers = None
    has_xformers = False

__all__ = ["FiLM", "GatedSkip", "Upsample", "Downsample", "ResBlock", "AttentionBlock"]


class GatedSkip(nn.Module):
    """
    A gated skip connection module that projects the encoder skip feature
    to match the decoder’s feature dimensions and learns a gating mask.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.gate = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, encoder_feat: torch.Tensor, decoder_feat: torch.Tensor) -> torch.Tensor:
        # Project the encoder feature to match decoder feature dimensions.
        proj_feat = self.proj(encoder_feat)
        # Compute a gating weight (values between 0 and 1)
        gate = torch.sigmoid(self.gate(encoder_feat))
        # Fuse the skip information into the decoder features.
        return decoder_feat + gate * proj_feat


class FiLM(nn.Module):
    def __init__(self, feature_dim, context_dim, pool: str = "mean"):
        """
        feature_dim: channels to modulate in the conv block
        context_dim: token dim (128)
        pool: 'mean' or 'attn' (mean is cheap & works well)
        """
        super().__init__()
        self.pool = pool
        if pool == "attn":
            self.query = nn.Linear(feature_dim, context_dim)
            self.key   = nn.Linear(context_dim, context_dim)
            self.value = nn.Linear(context_dim, context_dim)

        # Map pooled 128-d context to gamma/beta for the target feature_dim
        self.gamma_layer = nn.Linear(context_dim, feature_dim)
        self.beta_layer  = nn.Linear(context_dim, feature_dim)

    def forward(self, x, context):
        """
        x: [B, C, H, W, D]
        context: [B, L, context_dim] tokens
        """
        if self.pool == "mean":
            g = reduce(context, "b l c -> b c", "mean")  # [B, 128]
        else:
            # single-step cross-attn: query from x's global summary
            q = self.query(x.mean(dim=(2,3,4)))          # [B, 128]
            attn = torch.softmax((q @ self.key(context).transpose(1,2)) / (context.size(-1) ** 0.5), dim=-1)
            g = attn @ self.value(context)               # [B, 128]

        gamma = self.gamma_layer(g).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1,1]
        beta  = self.beta_layer(g).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class Upsample(nn.Module):
    """
    Convolution-based upsampling layer.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels to the layer.
        use_convtranspose: if True, use ConvTranspose to upsample feature maps in decoder.
    """

    def __init__(self, spatial_dims: int, in_channels: int, use_convtranspose: bool) -> None:
        super().__init__()
        if use_convtranspose:
            self.conv = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=in_channels,
                strides=2,
                kernel_size=3,
                padding=1,
                conv_only=True,
                is_transposed=True,
            )
        else:
            self.conv = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=in_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        self.use_convtranspose = use_convtranspose

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_convtranspose:
            return self.conv(x)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = x.dtype
        if dtype == torch.bfloat16:
            x = x.to(torch.float32)

        x = F.interpolate(x, scale_factor=2.0, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            x = x.to(dtype)

        x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    Convolution-based downsampling layer.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels.
    """

    def __init__(self, spatial_dims: int, in_channels: int) -> None:
        super().__init__()
        self.pad = (0, 1) * spatial_dims

        self.conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=2,
            kernel_size=3,
            padding=0,
            conv_only=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, self.pad, mode="constant", value=0.0)
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    """
    Residual block consisting of a cascade of 2 convolutions + activation + normalisation block, and a
    residual connection between input and output.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: input channels to the layer.
        norm_num_groups: number of groups involved for the group normalisation layer. Ensure that your number of
            channels is divisible by this number.
        norm_eps: epsilon for the normalisation.
        out_channels: number of output channels.
    """

    def __init__(self,
                 spatial_dims: int,
                 in_channels: int,
                 norm_num_groups: int,
                 norm_eps: float,
                 out_channels: int,
                 context_dim: int | None = None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        if context_dim is not None:
            self.film1 = FiLM(feature_dim=in_channels, context_dim=context_dim, pool="mean")

        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )
        self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=out_channels, eps=norm_eps, affine=True)
        if context_dim is not None:
            self.film2 = FiLM(feature_dim=self.out_channels, context_dim=context_dim, pool="mean")

        self.conv2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        if self.in_channels != self.out_channels:
            self.nin_shortcut = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        if context is not None:
            h = self.film1(h, context)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        if context is not None:
            h = self.film2(h, context)
        h = F.silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class AttentionBlock(nn.Module):
    """
    Attention block.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        num_channels: number of input channels.
        num_head_channels: number of channels in each attention head.
        norm_num_groups: number of groups involved for the group normalisation layer. Ensure that your number of
            channels is divisible by this number.
        norm_eps: epsilon value to use for the normalisation.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        num_head_channels: int | None = None,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.use_flash_attention = use_flash_attention
        self.spatial_dims = spatial_dims
        self.num_channels = num_channels

        self.num_heads = num_channels // num_head_channels if num_head_channels is not None else 1
        self.scale = 1 / math.sqrt(num_channels / self.num_heads)

        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels, eps=norm_eps, affine=True)

        self.to_q = nn.Linear(num_channels, num_channels)
        self.to_k = nn.Linear(num_channels, num_channels)
        self.to_v = nn.Linear(num_channels, num_channels)

        self.proj_attn = nn.Linear(num_channels, num_channels)

    def reshape_heads_to_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Divide hidden state dimension to the multiple attention heads and reshape their input as instances in the batch.
        """
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, dim // self.num_heads)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len, dim // self.num_heads)
        return x

    def reshape_batch_dim_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combine the output of the attention heads back into the hidden state dimension."""
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size // self.num_heads, self.num_heads, seq_len, dim)
        x = x.permute(0, 2, 1, 3).reshape(batch_size // self.num_heads, seq_len, dim * self.num_heads)
        return x

    def _memory_efficient_attention_xformers(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        x = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)
        return x

    def _attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1)
        x = torch.bmm(attention_probs, value)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        batch = channel = height = width = depth = -1
        if self.spatial_dims == 2:
            batch, channel, height, width = x.shape
        if self.spatial_dims == 3:
            batch, channel, height, width, depth = x.shape

        # norm
        x = self.norm(x)

        if self.spatial_dims == 2:
            x = x.view(batch, channel, height * width).transpose(1, 2)
        if self.spatial_dims == 3:
            x = x.view(batch, channel, height * width * depth).transpose(1, 2)

        # proj to q, k, v
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        # Multi-Head Attention
        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if self.use_flash_attention:
            x = self._memory_efficient_attention_xformers(query, key, value)
        else:
            x = self._attention(query, key, value)

        x = self.reshape_batch_dim_to_heads(x)
        x = x.to(query.dtype)

        if self.spatial_dims == 2:
            x = x.transpose(-1, -2).reshape(batch, channel, height, width)
        if self.spatial_dims == 3:
            x = x.transpose(-1, -2).reshape(batch, channel, height, width, depth)

        return x + residual

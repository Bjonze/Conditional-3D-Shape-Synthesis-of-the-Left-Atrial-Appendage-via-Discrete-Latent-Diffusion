"""Descriptor conditioning for the 3D VQ-GAN.

The 18 anatomical descriptors are mapped to a sequence of context tokens that
FiLM-modulate every residual block of the encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ScalarTokenizer"]


class ScalarTokenizer(nn.Module):
    """
    Maps 18 scalars -> 18 tokens of dim 128.
    - Shared MLP for all scalars (good bias + small params)
    - Per-feature learnable ID embeddings
    - Optional tiny Transformer encoder for cross-feature mixing
    """
    def __init__(
        self,
        num_features: int = 18,
        token_dim: int = 128,
        hidden: int = 32,
        use_transformer: bool = True,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_mult: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Shared scalar → token
        self.scalar_mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, token_dim),
            nn.LayerNorm(token_dim),
        )

        # Per-feature ID embedding
        self.feature_emb = nn.Embedding(num_features, token_dim)
        self.ctx_bos = nn.Parameter(torch.zeros(1, 1, token_dim))

        self.use_transformer = use_transformer
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=token_dim, nhead=n_heads,
                dim_feedforward=ff_mult * token_dim,
                dropout=dropout, batch_first=True, activation="gelu",
                norm_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # small init for stability
        nn.init.trunc_normal_(self.feature_emb.weight, std=0.02)

    def forward(self, x):
        """
        x: [B, F, 1] continuous scalars (recommended: z-scored per feature)
        returns: [B, F, 128]
        """
        B, F, D = x.shape
        assert D == 1, "Expected each feature to be a scalar."

        # shared scalar→token
        tokens = self.scalar_mlp(x)                      # [B, F, 128]
        # add feature IDs
        ids = torch.arange(F, device=x.device).unsqueeze(0).expand(B, F)
        tokens = tokens + self.feature_emb(ids)          # [B, F, 128]

        bos = self.ctx_bos.expand(B, 1, -1)                   # (B, 1, d_model)
        tokens = torch.cat([bos, tokens], dim=1)                        # (B, F+1, d_model)

        if self.use_transformer:
            tokens = self.transformer(tokens)            # [B, F, 128]

        return tokens

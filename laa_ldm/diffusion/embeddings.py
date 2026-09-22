"""Embeddings used by the discrete latent diffusion model.

Two of them:

* :class:`ScalarTokenizer` turns the 18 continuous shape descriptors into a
  sequence of conditioning tokens that the denoising transformer cross-attends
  to.
* :class:`MaskedTokenEmbedding3D` embeds the 8x8x8 grid of codebook indices,
  with one extra entry for the ``[MASK]`` token of the diffusion process and
  factorised depth/height/width position embeddings.
"""

import torch
import torch.nn as nn

__all__ = ["BaseEmbedding", "ScalarTokenizer", "MaskedTokenEmbedding3D"]


class BaseEmbedding(nn.Module):
    """Embedding base class that can be frozen through ``trainable=False``."""

    def get_loss(self):
        return None

    def forward(self, **kwargs):
        raise NotImplementedError

    def train(self, mode=True):
        self.training = mode
        if self.trainable and mode:
            super().train()
        return self

    def _set_trainable(self):
        if not self.trainable:
            for _, p in self.named_parameters():
                p.requires_grad = False
            self.eval()


class ScalarTokenizer(nn.Module):
    """Map ``num_features`` scalars to ``num_features + 1`` conditioning tokens.

    A shared MLP lifts each scalar to ``token_dim``, a learnable per-feature ID
    embedding tells the model which descriptor it is looking at, a BOS token is
    prepended, and a small transformer encoder mixes the features.  The output
    is L2-normalised when ``normalize`` is set, which keeps the scale of the
    conditioning signal stable across descriptors.

    Args:
        num_features: number of input scalars (18 descriptors).
        token_dim: width of the produced tokens.
        hidden: width of the shared scalar MLP.
        use_transformer: apply the cross-feature transformer encoder.
        n_heads, n_layers, ff_mult, dropout: transformer encoder settings.
        normalize: L2-normalise the output tokens.

    Shape:
        input ``(B, F)`` or ``(B, F, 1)`` -> output ``(B, F + 1, token_dim)``.
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
        normalize: bool = True,
    ):
        super().__init__()
        self.normalize = normalize
        self.embed_dim = token_dim

        self.scalar_mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, token_dim),
            nn.LayerNorm(token_dim),
        )

        self.feature_emb = nn.Embedding(num_features, token_dim)
        self.ctx_bos = nn.Parameter(torch.zeros(1, 1, token_dim))

        self.use_transformer = use_transformer
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=n_heads,
                dim_feedforward=ff_mult * token_dim,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        nn.init.trunc_normal_(self.feature_emb.weight, std=0.02)
        self.trainable = True
        self._set_trainable()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, F, 1)
        B, F, D = x.shape
        assert D == 1, "Expected each feature to be a scalar."

        tokens = self.scalar_mlp(x)                          # (B, F, token_dim)
        ids = torch.arange(F, device=x.device).unsqueeze(0).expand(B, F)
        tokens = tokens + self.feature_emb(ids)

        bos = self.ctx_bos.expand(B, 1, -1)
        tokens = torch.cat([bos, tokens], dim=1)             # (B, F + 1, token_dim)

        if self.use_transformer:
            tokens = self.transformer(tokens)

        if self.normalize:
            tokens = tokens / tokens.norm(dim=-1, keepdim=True)

        return tokens

    def train(self, mode=True):
        self.training = mode
        if self.trainable and mode:
            super().train()
        return self

    def _set_trainable(self):
        if not self.trainable:
            for _, p in self.named_parameters():
                p.requires_grad = False
            self.eval()

    def get_loss(self):
        return None


class MaskedTokenEmbedding3D(BaseEmbedding):
    """Embed a flattened 3D grid of codebook indices, plus the ``[MASK]`` token.

    Position information is factorised over the three axes: one embedding per
    depth, height and width index, summed.

    Args:
        num_embed: codebook size; one extra entry is added for ``[MASK]``.
        spatial_size: latent grid as ``(depth, height, width)``.
        embed_dim: transformer width.
        pos_emb_type: ``"embedding"`` (nn.Embedding) or ``"parameter"``.
    """

    def __init__(self,
                 num_embed=512,
                 spatial_size=[8, 8, 8],
                 embed_dim=512,
                 trainable=True,
                 pos_emb_type='embedding'):
        super().__init__()

        if isinstance(spatial_size, int):
            spatial_size = [spatial_size, spatial_size, spatial_size]

        self.spatial_size = spatial_size
        self.num_embed = num_embed + 1  # +1 for [MASK]
        self.embed_dim = embed_dim
        self.trainable = trainable
        self.pos_emb_type = pos_emb_type

        assert self.pos_emb_type in ['embedding', 'parameter']

        self.emb = nn.Embedding(self.num_embed, embed_dim)
        if self.pos_emb_type == 'embedding':
            self.depth_emb = nn.Embedding(self.spatial_size[0], embed_dim)
            self.height_emb = nn.Embedding(self.spatial_size[1], embed_dim)
            self.width_emb = nn.Embedding(self.spatial_size[2], embed_dim)
        else:
            self.depth_emb = nn.Parameter(torch.zeros(1, self.spatial_size[0], embed_dim))
            self.height_emb = nn.Parameter(torch.zeros(1, self.spatial_size[1], embed_dim))
            self.width_emb = nn.Parameter(torch.zeros(1, self.spatial_size[2], embed_dim))

        self._set_trainable()

    def forward(self, index, **kwargs):
        assert index.dim() == 2  # B x L
        try:
            index[index < 0] = 0
            emb = self.emb(index)
        except Exception:
            raise RuntimeError('IndexError: index out of range in self, max index {}, num embed {}'
                               .format(index.max(), self.num_embed))

        if emb.shape[1] > 0:
            d, h, w = self.spatial_size
            if self.pos_emb_type == 'embedding':
                device = index.device
                depth_emb = self.depth_emb(torch.arange(d, device=device).view(1, d)).unsqueeze(2).unsqueeze(3)
                height_emb = self.height_emb(torch.arange(h, device=device).view(1, h)).unsqueeze(1).unsqueeze(3)
                width_emb = self.width_emb(torch.arange(w, device=device).view(1, w)).unsqueeze(1).unsqueeze(2)
            else:
                depth_emb = self.depth_emb.unsqueeze(2).unsqueeze(3)
                height_emb = self.height_emb.unsqueeze(1).unsqueeze(3)
                width_emb = self.width_emb.unsqueeze(1).unsqueeze(2)
            pos_emb = (depth_emb + height_emb + width_emb).view(1, d * h * w, -1)
            emb = emb + pos_emb[:, :emb.shape[1], :]

        return emb

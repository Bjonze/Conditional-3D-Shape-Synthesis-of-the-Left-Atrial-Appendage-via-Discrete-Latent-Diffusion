"""The denoising transformer that predicts p(x_0 | x_t, descriptors).

Each block is self-attention over the 512 latent tokens followed by
cross-attention to the descriptor tokens.  The diffusion timestep enters
through adaptive layer norm (AdaLN), so the same weights serve every step.

Adapted from VQ-Diffusion (Microsoft, MIT licence); the variants for text and
class conditioning were removed, leaving the self+cross attention path used in
the paper.
"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint

from laa_ldm.utils.config import instantiate_from_config

__all__ = ["DenoisingTransformer", "Block", "FullAttention", "CrossAttention", "AdaLayerNorm"]


class FullAttention(nn.Module):
    """Non-causal multi-head self-attention over the latent tokens."""

    def __init__(self,
                 n_embd, # the embed dim
                 n_head, # the number of heads
                 seq_len=None, # the max length of sequence
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class CrossAttention(nn.Module):
    """Multi-head attention from latent tokens to the conditioning tokens."""

    def __init__(self,
                 condition_seq_len,
                 n_embd, # the embed dim
                 condition_embd, # condition dim
                 n_head, # the number of heads
                 seq_len=None, # the max length of sequence
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class GELU2(nn.Module):
    """Sigmoid-approximated GELU used by VQ-Diffusion."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(1.702 * x)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding of the (rescaled) diffusion timestep."""

    def __init__(self, num_steps, dim, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class AdaLayerNorm(nn.Module):
    """LayerNorm whose scale and shift are predicted from the diffusion timestep."""

    def __init__(self, n_embd, diffusion_step, emb_type="adalayernorm_abs"):
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(diffusion_step, n_embd)
        else:
            self.emb = nn.Embedding(diffusion_step, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)
        self.diff_step = diffusion_step

    def forward(self, x, timestep):
        if timestep[0] >= self.diff_step:
            _emb = self.emb.weight.mean(dim=0, keepdim=True).repeat(len(timestep), 1)
            emb = self.linear(self.silu(_emb)).unsqueeze(1)
        else:
            emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class Block(nn.Module):
    """Self-attention over latent tokens, then cross-attention to the condition."""

    def __init__(self,
                 condition_seq_len=19,
                 n_embd=512,
                 n_head=4,
                 seq_len=512,
                 attn_pdrop=0.0,
                 resid_pdrop=0.0,
                 mlp_hidden_times=4,
                 activate='GELU2',
                 attn_type='selfcross',
                 condition_dim=128,
                 diffusion_step=100,
                 timestep_type='adalayernorm',
                 ):
        super().__init__()
        self.attn_type = attn_type
        if attn_type != 'selfcross':
            raise ValueError(f"Unsupported attn_type {attn_type!r}; expected 'selfcross'.")
        if 'adalayernorm' not in timestep_type:
            raise ValueError(f"Unsupported timestep_type {timestep_type!r}.")
        if activate not in ('GELU', 'GELU2'):
            raise ValueError(f"Unsupported activation {activate!r}.")

        self.ln1 = AdaLayerNorm(n_embd, diffusion_step, timestep_type)
        self.ln1_1 = AdaLayerNorm(n_embd, diffusion_step, timestep_type)
        self.ln2 = nn.LayerNorm(n_embd)

        self.attn1 = FullAttention(
            n_embd=n_embd,
            n_head=n_head,
            seq_len=seq_len,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )
        self.attn2 = CrossAttention(
            condition_seq_len,
            n_embd=n_embd,
            condition_embd=condition_dim,
            n_head=n_head,
            seq_len=seq_len,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )

        act = nn.GELU() if activate == 'GELU' else GELU2()
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, encoder_output, timestep, mask=None):
        a, att = self.attn1(self.ln1(x, timestep), encoder_output, mask=mask)
        x = x + a
        a, att = self.attn2(self.ln1_1(x, timestep), encoder_output, mask=mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, att


class DenoisingTransformer(nn.Module):
    """Stack of self+cross attention blocks predicting logits over the codebook.

    Args:
        condition_seq_len: number of conditioning tokens (18 descriptors + BOS).
        n_layer, n_embd, n_head: transformer size.
        content_seq_len: number of latent tokens (8*8*8 = 512).
        content_spatial_size: latent grid, used by the content embedding.
        condition_dim: width of the conditioning tokens.
        diffusion_step: number of diffusion steps, sets the AdaLN time table.
        content_emb_config: config of the content (token) embedding.

    Shape:
        ``(B, L)`` indices -> ``(B, num_codes, L)`` logits.
    """

    def __init__(
        self,
        condition_seq_len=19,
        n_layer=8,
        n_embd=512,
        n_head=4,
        content_seq_len=512,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        mlp_hidden_times=2,
        block_activate='GELU2',
        attn_type='selfcross',
        content_spatial_size=(8, 8, 8),
        condition_dim=128,
        diffusion_step=100,
        timestep_type='adalayernorm',
        content_emb_config=None,
        checkpoint=False,
    ):
        super().__init__()

        self.use_checkpoint = checkpoint
        self.content_emb = instantiate_from_config(content_emb_config)
        self.content_spatial_size = content_spatial_size

        self.blocks = nn.Sequential(*[Block(
            condition_seq_len,
            n_embd=n_embd,
            n_head=n_head,
            seq_len=content_seq_len,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            mlp_hidden_times=mlp_hidden_times,
            activate=block_activate,
            attn_type=attn_type,
            condition_dim=condition_dim,
            diffusion_step=diffusion_step,
            timestep_type=timestep_type,
        ) for _ in range(n_layer)])

        # Prediction head: one logit per codebook entry, excluding [MASK].
        out_cls = self.content_emb.num_embed - 1
        self.to_logits = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, out_cls),
        )

        self.condition_seq_len = condition_seq_len
        self.content_seq_len = content_seq_len

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Normal(0, 0.02) for linear/embedding weights, identity for LayerNorm."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine == True:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(self, input, cond_emb, t):
        emb = self.content_emb(input)

        for block_idx in range(len(self.blocks)):   
            if self.use_checkpoint:
                emb, att_weight = checkpoint(self.blocks[block_idx], emb, cond_emb, t)
            else:
                emb, att_weight = self.blocks[block_idx](emb, cond_emb, t)
        logits = self.to_logits(emb)  # (B, L, num_codes)
        out = rearrange(logits, 'b l c -> b c l')
        return out

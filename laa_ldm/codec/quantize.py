"""EMA vector quantiser for the 3D VQ-GAN.

The codebook is updated with exponential moving averages rather than gradients.
Codes are matched by cosine distance (``distance_metric: cosine``), which keeps
the codebook on the unit sphere and avoids the index collapse we observed with
plain L2 on binary masks.  Dead codes are periodically re-seeded from a buffer
of recent encoder outputs (:meth:`EmbeddingEMA.refresh_codes`).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ["EmbeddingEMA", "EMAVectorQuantizer3D"]


class EmbeddingEMA(nn.Module):
    """Codebook whose entries are tracked with exponential moving averages."""

    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.decay = decay
        self.eps = eps        
        weight = torch.randn(num_tokens, codebook_dim)
        self.weight = nn.Parameter(weight, requires_grad = False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad = False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad = False)
        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg): 
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
            )
        #normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)
    
    def refresh_codes(self, latent_buffer, thresh=1.0):
        dead = (self.cluster_size < thresh).nonzero(as_tuple=True)[0]
        if dead.numel() and latent_buffer is not None:
            idx = torch.randint(0, latent_buffer.size(0), (dead.numel(),), device=latent_buffer.device)
            self.weight.data[dead] = F.normalize(latent_buffer[idx], dim=1)


class EMAVectorQuantizer3D(nn.Module):
    """Quantise a (B, C, D, H, W) feature map against an EMA-updated codebook.

    Args:
        n_e: codebook size (number of entries).
        e_dim: codebook dimension, must equal the channel count of ``z``.
        beta: weight of the commitment loss.
        decay: EMA decay for the cluster sizes and code averages.
        distance_metric: ``"cosine"`` (used in the paper) or ``"l2"``.
    """

    def __init__(self, n_e, e_dim, beta, decay=0.99, eps=1e-5, distance_metric="cosine"):
        super().__init__()
        self.num_tokens = n_e
        self.codebook_dim = e_dim
        self.beta = beta
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps)
        self.distance_metric = distance_metric

        if distance_metric not in ["l2", "cosine"]:
            raise ValueError(f"Unknown distance_metric: {distance_metric}, must be 'l2' or 'cosine'.")

    def forward(self, z):
        """Returns ``(z_q, commitment_loss, (perplexity, encodings, indices))``."""
        # (B, C, D, H, W) -> (B, D, H, W, C), then flatten the spatial axes
        z = rearrange(z, 'b c d h w -> b d h w c').contiguous()
        z_flattened = z.reshape(-1, self.codebook_dim)  # (N, C)
        with torch.cuda.amp.autocast(enabled=False):
            z_flattened = z_flattened.float()
            weight = self.embedding.weight.float()

            if self.distance_metric == "l2":
                d = (z_flattened.pow(2).sum(dim=1, keepdim=True)
                    + weight.pow(2).sum(dim=1)
                    - 2 * torch.einsum('bd,nd->bn', z_flattened, weight))
            elif self.distance_metric == "cosine":
                zf = F.normalize(z_flattened, dim=1)
                ew = F.normalize(weight, dim=1)
                d = - (zf @ ew.t())
            else:
                raise ValueError(f"Unknown distance_metric: {self.distance_metric}")

        encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(encoding_indices).view(z.shape)
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)
        encodings_sum = encodings.sum(0)
        embed_input = zf if self.distance_metric == "cosine" else z_flattened

        if self.training and self.embedding.update:
            # EMA statistics are summed across ranks so every replica keeps an
            # identical codebook under DDP.
            embed_sum = encodings.transpose(0, 1) @ embed_input
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(encodings_sum)
                torch.distributed.all_reduce(embed_sum)
            self.embedding.cluster_size_ema_update(encodings_sum)
            self.embedding.embed_avg_ema_update(embed_sum)
            self.embedding.weight_update(self.num_tokens)
            if self.distance_metric == "cosine":
                # Keep weights on the unit sphere for consistent cosine similarity
                self.embedding.weight.data = F.normalize(self.embedding.weight.data, dim=1)

        avg_probs = encodings_sum / encodings_sum.sum().clamp_min(1.0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        loss = self.beta * F.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b d h w c -> b c d h w')
        return z_q, loss, (perplexity, encodings, encoding_indices)

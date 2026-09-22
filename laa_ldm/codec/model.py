"""The VQ-GAN codec: a conditional 3D VQ-GAN over binary LAA masks.

Stage 1 of the pipeline.  A 128^3 mask is encoded to an 8x8x8 grid of discrete
codes (512-entry codebook), conditioned on the 18 shape descriptors through
FiLM layers in the encoder, and decoded back to mask logits.  Training uses
manual optimisation: one optimiser for the autoencoder, one for the patch
discriminator, with gradient accumulation to reach the configured effective
batch size.
"""

from __future__ import annotations

import math
import os
import random
from collections.abc import Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from monai.utils import ensure_tuple_rep
from torch.nn import Conv3d
from torch.optim.lr_scheduler import LambdaLR

from laa_ldm.codec.conditioning import ScalarTokenizer
from laa_ldm.codec.losses import VQGANLoss
from laa_ldm.codec.metrics import CosineAnnealingWarmup, compute_metrics
from laa_ldm.codec.nets import Decoder, Encoder
from laa_ldm.codec.quantize import EMAVectorQuantizer3D
from laa_ldm.utils.mesh import save_mesh

__all__ = ["VQGAN3D"]


class VQGAN3D(pl.LightningModule):
    """Conditional 3D VQ-GAN over binary LAA masks."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int = 1,
        out_channels: int = 1,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        num_channels: Sequence[int] = (32, 64, 256), #128, 256
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
        batch_size: int = 1,
        E_batch_size: int = 1,
        dataset_len: int = 1,
        n_epochs: int = 1,
        num_warmup_epochs: int = 1,
        beta: float = 0.25,
        ema_decay: float = 0.99,
        distance_metric: str = "l2",
        lambda_entropy: float = 1e-4,
        lr_ae: float = 1e-4,
        lr_disc: float = 5e-5,
        disc_cfg: dict | None = None,
        losses_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        # All number of channels should be multiple of num_groups
        if any((out_channel % norm_num_groups) != 0 for out_channel in num_channels):
            raise ValueError("AutoencoderKL expects all num_channels being multiple of norm_num_groups")

        if len(num_channels) != len(attention_levels):
            raise ValueError("AutoencoderKL expects num_channels being same size of attention_levels")

        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(num_channels))

        if len(num_res_blocks) != len(num_channels):
            raise ValueError(
                "`num_res_blocks` should be a single integer or a tuple of integers with the same length as "
                "`num_channels`."
            )
        disc_cfg   = dict(disc_cfg or {})
        losses_cfg = dict(losses_cfg or {})

        self.save_hyperparameters()
        self.automatic_optimization = False  # manual optimization (for multiple optimizers)

        self.in_channels = in_channels
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.dataset_len = int(dataset_len)
        self.per_device_batch_size = int(batch_size)
        self.target_effective_batch_size = int(E_batch_size)
        self.n_epochs = int(n_epochs)
        self.num_warmup_epochs = int(num_warmup_epochs)
        self.world_size = 1
        self.accumulate_batches = 1
        self.steps_per_epoch = 1
        self.total_steps = 1
        self.warmup_steps = 0
        self._configure_training_schedule()
        self.gradient_accumulation_steps = 0

        self.beta = beta
        self.ema_decay = ema_decay
        self.lambda_entropy = float(lambda_entropy)
        self.lr_ae = lr_ae
        self.lr_disc = lr_disc

        # Instantiate sub-networks
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
            context_dim=context_dim
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
        self.quant_conv = Conv3d(
            in_channels=codebook_dim,
            out_channels=codebook_dim,
            stride=1,
            kernel_size=1,
            padding=0,
        )
        self.post_quant_conv = Conv3d(
            in_channels=codebook_dim,
            out_channels=codebook_dim,
            stride=1,
            kernel_size=1,
            padding=0,
        )
        self.quantize = EMAVectorQuantizer3D(codebook_size, codebook_dim, beta,
                                             decay=self.ema_decay, distance_metric=distance_metric)
        self.context_projector = ScalarTokenizer(num_features=num_context_vars,
                                                 token_dim=context_dim,
                                                 hidden = 32,
                                                 use_transformer = True,
                                                 n_heads = 4,
                                                 n_layers = 1,
                                                 ff_mult = 2,
                                                 dropout = 0.0)

        self.disc_start  = int(disc_cfg.get("start", 1000))
        self.disc_ndf    = int(disc_cfg.get("ndf", 64))
        self.disc_weight = float(disc_cfg.get("weight", 0.8))
        self.codebook_weight = float(losses_cfg.get("codebook_weight", 1.0))
        self.r1_gamma = float(disc_cfg.get("r1_gamma", 10))
        self.init_noise_std = float(disc_cfg.get("init_noise_std", 0.1))
        self.noise_decay_steps = float(disc_cfg.get("noise_decay_steps", 0.2))
        self.noise_decay_steps = int(self.noise_decay_steps * self.total_steps) if self.noise_decay_steps is None else int(self.noise_decay_steps)

        # composite loss, configured entirely from the YAML
        self.loss = VQGANLoss(
            disc_start=self.disc_start,
            disc_in_channels=int(disc_cfg.get("in_channels", 1)),
            disc_num_layers=int(disc_cfg.get("n_layers", 3)),
            disc_ndf=self.disc_ndf,
            disc_loss=str(disc_cfg.get("loss", "hinge")),
            use_actnorm=bool(disc_cfg.get("use_actnorm", False)),
            disc_factor=float(disc_cfg.get("factor", 1.0)),
            disc_weight=self.disc_weight,
            codebook_weight=self.codebook_weight,
            perceptual_weight=float(losses_cfg.get("perceptual_weight", 0.0)),
            r1_gamma=self.r1_gamma,
            disc_conditional=False,
        )
        self.num_steps = 0

    def _current_world_size(self) -> int:
        trainer = getattr(self, "_trainer", None)
        if trainer is not None:
            try:
                return max(1, int(trainer.world_size))
            except (TypeError, ValueError):
                pass
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return max(1, int(torch.distributed.get_world_size()))
        return 1

    def _configure_training_schedule(self) -> None:
        self.world_size = self._current_world_size()
        micro_batch = max(1, self.per_device_batch_size * self.world_size)
        self.accumulate_batches = max(1, math.ceil(self.target_effective_batch_size / micro_batch))
        actual_effective_batch = micro_batch * self.accumulate_batches
        num_batches = getattr(getattr(self, "_trainer", None), "num_training_batches", None)
        if isinstance(num_batches, int):
            self.steps_per_epoch = max(1, math.ceil(num_batches / self.accumulate_batches))
        elif isinstance(num_batches, float) and math.isfinite(num_batches):
            self.steps_per_epoch = max(1, math.ceil(int(num_batches) / self.accumulate_batches))
        else:
            self.steps_per_epoch = max(1, math.ceil(self.dataset_len / actual_effective_batch))
        self.total_steps = max(1, self.n_epochs * self.steps_per_epoch)
        self.warmup_steps = max(0, math.ceil(self.num_warmup_epochs * self.steps_per_epoch))

    def _should_step_optimizer(self, batch_idx: int) -> bool:
        accum = max(1, int(self.accumulate_batches))
        if (batch_idx + 1) % accum == 0:
            return True

        num_batches = getattr(getattr(self, "_trainer", None), "num_training_batches", None)
        if isinstance(num_batches, int):
            return (batch_idx + 1) >= num_batches
        if isinstance(num_batches, float) and math.isfinite(num_batches):
            return (batch_idx + 1) >= int(num_batches)
        return False

    def _refresh_codes(self, latent_buffer: torch.Tensor, thresh: float = 1.0) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                self.quantize.embedding.refresh_codes(latent_buffer, thresh=thresh)
            torch.distributed.broadcast(self.quantize.embedding.weight.data, src=0)
            return
        self.quantize.embedding.refresh_codes(latent_buffer, thresh=thresh)

    def encode(self, x, cond):
        if x.dim() == 6:
            x = x.squeeze(1)
        if x.dim() == 4:
            x = x.unsqueeze(1)
        ctx = self.context_projector(cond)  # Project context variables
        h = self.encoder(x, ctx)
        h = self.quant_conv(h)
        if self.training:
            if not hasattr(self, "latent_buffer"):
                self.latent_buffer = []
            hb = h.detach().flatten(2).transpose(1, 2).reshape(-1, h.size(1))  # (N, C)
            self.latent_buffer.append(hb)
            if len(self.latent_buffer) > 50:
                self.latent_buffer.pop(0)

        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info
    
    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec
    
    def forward(self, x, cond):
        """Full forward: encode->quantize->decode. Used for inference/validation."""
        if x.dim() == 6:
            x = x.squeeze(1)
        quant, emb_loss, _ = self.encode(x, cond)                   # Encoder with FiLM conditioning
        dec = self.decode(quant)
        return dec, emb_loss

    def training_step(self, batch, batch_idx):    
        opt_ae, opt_disc = self.optimizers()  # get generator & discriminator optimizers
        self.train()
        images, clip_context = batch
        if images.dim() == 6:
            images = images.squeeze()
        # Generator forward
        accum = max(1, int(self.accumulate_batches))
        do_step = self._should_step_optimizer(batch_idx)

        # ---------------- G / AE ----------------
        with opt_ae.toggle_model(sync_grad=do_step):
            quant, qloss, info = self.encode(images, clip_context)
            xrec = self.decode(quant)

            aeloss, log_dict_ae = self.loss(qloss, images, xrec, 0, self.num_steps,
                                            last_layer=self.get_last_layer(), split="train")
            
            encoding_indices = info[2]                 # (B*D*H*W,)
            counts = torch.bincount(encoding_indices, minlength=self.codebook_size).float()
            probs  = counts / (counts.sum() + 1e-8)
            entropy = -(probs * (probs.clamp_min(1e-8)).log()).sum()

            aeloss = aeloss + (-self.lambda_entropy * entropy)   # encourage higher usage entropy

            aeloss = aeloss / accum                        # keep effective LR the same
            self.manual_backward(aeloss)
            if do_step:
                self.clip_gradients(opt_ae, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
                opt_ae.step()
                opt_ae.zero_grad(set_to_none=True)
                scheduler = self.lr_schedulers()
                scheduler.step()
                self.log("train/lr_ae", scheduler.get_last_lr()[0], prog_bar=True, on_step=True, on_epoch=False)

        # ---------------- D ----------------
        noise_std = max(0.0, self.init_noise_std * (1.0 - float(self.gradient_accumulation_steps) / max(1, self.noise_decay_steps)))
        noisy_real = images
        noisy_fake = xrec.detach()
        if noise_std > 0.0:
            noisy_real = noisy_real + torch.randn_like(noisy_real) * noise_std
            noisy_fake = noisy_fake + torch.randn_like(noisy_fake) * noise_std

        with opt_disc.toggle_model(sync_grad=do_step):
            discloss, log_dict_disc = self.loss(
                qloss, noisy_real, noisy_fake, optimizer_idx=1, global_step=self.num_steps,
                last_layer=self.get_last_layer(), split="train"
            )
            discloss = discloss / accum                      # keep effective LR the same
            self.manual_backward(discloss)
            if do_step:
                self.clip_gradients(opt_disc, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
                opt_disc.step()
                opt_disc.zero_grad(set_to_none=True)
                if self.num_steps > self.disc_start:
                    self.gradient_accumulation_steps += 1

        if self.training and (self.num_steps % 2000) == 0 and getattr(self, "latent_buffer", None):
            buf = torch.cat(self.latent_buffer, dim=0)   # (M, C)
            self._refresh_codes(buf, thresh=5.0)

        # logging
        self.log("train/aeloss", aeloss * accum, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/discloss", discloss * accum, prog_bar=True, on_step=True, on_epoch=False)
        self.log_dict({**log_dict_ae, **{"train/perplexity": info[0]}}, on_step=True, on_epoch=False)
        self.log_dict(log_dict_disc, on_step=True, on_epoch=False)
        self.num_steps += 1

    
    def validation_step(self, batch, batch_idx):
        self.eval()
        images, clip_context = batch
        if images.dim() == 6:
            images = images.squeeze()
        if images.dim() == 4:
            images = images.unsqueeze(1)

        quant, qloss, info = self.encode(images, clip_context)
        xrec = self.decode(quant)

        _, log_dict_ae = self.loss(qloss, images, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        _, log_dict_disc = self.loss(qloss, images, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        precision, recall, f1, iou = compute_metrics(torch.sigmoid(xrec), images)
        log = {**log_dict_ae, **log_dict_disc,
               "val/perplexity": info[0], "val/precision": precision,
               "val/recall": recall, "val/f1": f1, "val/iou": iou}

        log = {
            key: value.detach().to(self.device) if isinstance(value, torch.Tensor)
            else torch.tensor(value, device=self.device, dtype=torch.float32)
            for key, value in log.items()
        }
        self.train()
        return self.log_dict(log, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
    
    def on_validation_epoch_end(self):
        # Save exactly 10 reconstructions once per whole validation epoch (rank 0 only)
        self.eval()
        if getattr(self, "images_dir", None) is None:
            return
        if int(getattr(self, "global_rank", 0)) != 0:
            return
        trainer = getattr(self, "_trainer", None)
        if trainer is None or getattr(trainer, "sanity_checking", False):
            return

        val_ds = getattr(trainer.datamodule, "val_ds", None)
        if val_ds is None:
            return

        try:
            os.makedirs(self.images_dir, exist_ok=True)
            out_root = os.path.join(self.images_dir, f"epoch_{self.current_epoch:03d}", "val_samples")
            os.makedirs(out_root, exist_ok=True)

            n = len(val_ds)
            k = 10 if n >= 10 else n
            inds = random.sample(range(n), k) if n >= 10 else [random.randrange(n) for _ in range(k)]

            imgs, ctxs, names = [], [], []
            npz_list = getattr(val_ds, "npz_files", None)  # if your dataset exposes file names
            for i in inds:
                item = val_ds[i]
                im, ctx_clip = item[0], item[1]
                imgs.append(im)
                ctxs.append(ctx_clip)
                if npz_list is not None and i < len(npz_list):
                    stem = os.path.splitext(npz_list[i])[0]
                else:
                    stem = f"idx_{i}"
                names.append(stem)

            device = self.device
            imgs = torch.stack(imgs, dim=0).to(device, non_blocking=True)
            ctxs = torch.stack(ctxs, dim=0).to(device, non_blocking=True)

            with torch.no_grad():
                q, _, _ = self.encode(imgs, ctxs)
                xhat = self.decode(q)

            xhat_sig = torch.sigmoid(xhat[:, 0]).detach().cpu().numpy()

            for b in range(xhat_sig.shape[0]):
                vol = np.asarray(xhat_sig[b], dtype=np.float32)
                if vol.ndim != 3:
                    vol = np.squeeze(vol)

                save_mesh(vol, os.path.join(out_root, f"{names[b]}.stl"), level=0.5)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[on_validation_epoch_end save error] {e}", flush=True)
        self.train()

    def configure_optimizers(self):
        self._configure_training_schedule()
        if getattr(getattr(self, "_trainer", None), "is_global_zero", True):
            actual_effective_batch = self.per_device_batch_size * self.world_size * self.accumulate_batches
            print(
                "[VQGAN3D] "
                f"world_size={self.world_size}, per_device_batch_size={self.per_device_batch_size}, "
                f"accumulate_batches={self.accumulate_batches}, "
                f"effective_batch_size={actual_effective_batch}, steps_per_epoch={self.steps_per_epoch}",
                flush=True,
            )

        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters())+
                                  list(self.context_projector.parameters()),
                                  lr=self.lr_ae, betas=(0.5, 0.9))
        
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=self.lr_disc, betas=(0.5, 0.9))

        lr_lambda = CosineAnnealingWarmup(num_steps=self.total_steps, num_warmup_steps=self.warmup_steps)
        scheduler = LambdaLR(opt_ae, lr_lambda=lr_lambda)

        return [opt_ae, opt_disc], [{"scheduler": scheduler, "interval": "step"}]
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight

"""Composite reconstruction loss for the 3D VQ-GAN.

The generator loss combines

* a positively weighted binary cross-entropy on the mask logits,
* an optional 2.5D LPIPS perceptual term (MONAI ``PerceptualLoss``), and
* a least-squares adversarial term from a 3D patch discriminator,

plus the codebook commitment loss produced by the quantiser.  The adversarial
term is switched on only after ``disc_start`` optimiser steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import PatchDiscriminator

__all__ = ["VQGANLoss", "adopt_weight", "hinge_d_loss", "vanilla_d_loss"]


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQGANLoss(nn.Module):
    """BCE (+ perceptual) reconstruction loss with a patch-GAN discriminator."""

    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=15.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge", r1_gamma=10.0):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.disc_in_channels = disc_in_channels
        self.perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="squeeze", #squeeze
                                            is_fake_3d=True, fake_3d_ratio=0.20) #0.2, is_fake_3d=True
        self.perceptual_weight = perceptual_weight
        self.register_buffer("bce_pos_weight", torch.tensor([40.0]))

        self.discriminator = PatchDiscriminator(spatial_dims=3, 
                                            num_layers_d=disc_num_layers, 
                                            num_channels=32, 
                                            in_channels=self.disc_in_channels, 
                                            out_channels=self.disc_in_channels)
        self.adv_loss_fn   = PatchAdversarialLoss(criterion="least_squares")

        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQGANLoss running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.r1_gamma = r1_gamma

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 64.0).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        if self.disc_in_channels == 1:
            rec_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                reconstructions.contiguous(),
                inputs.contiguous(),
                pos_weight=self.bce_pos_weight
            ) * self.pixel_weight if self.pixel_weight != 0 else torch.tensor(0.0, device=inputs.device)
        else:
            raise ValueError("VQGANLoss expects single-channel (mask) inputs.")
            
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss.clone()
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                gen_adv_loss = self.adv_loss_fn(self.discriminator(reconstructions.float())[-1], True, False)
            else:
                assert self.disc_conditional
                gen_adv_loss = self.adv_loss_fn(self.discriminator(reconstructions.float())[-1], True, False)
            #g_loss = -torch.mean(logits_fake)

            try:
                d_weight = torch.tensor(1.0, device=reconstructions.device) #self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0, device=reconstructions.device)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * gen_adv_loss + self.codebook_weight * codebook_loss.mean()

            log = {"{}/total_loss".format(split): loss.clone().detach().cpu().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().cpu().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().cpu().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().cpu().mean(),
                   "{}/p_loss".format(split): p_loss.detach().cpu().mean(),
                   "{}/d_weight".format(split): d_weight.detach().cpu().mean(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor).cpu().mean(),
                   "{}/g_loss".format(split): gen_adv_loss.detach().cpu().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # inputs/reconstructions may already include instance noise (from training_step)
            if cond is None:
                logits_fake = self.discriminator(reconstructions.contiguous().detach())[-1]
                logits_real = self.discriminator(inputs.contiguous())[-1]
                d_loss_fake = self.adv_loss_fn(logits_fake, False, True)
                d_loss_real = self.adv_loss_fn(logits_real, True, True)
                total_d_loss = 0.5 * (d_loss_fake + d_loss_real)
            else:
                logits_fake = self.discriminator(reconstructions.contiguous().detach())[-1]
                logits_real = self.discriminator(inputs.contiguous())[-1]
                d_loss_fake = self.adv_loss_fn(logits_fake, False, True)
                d_loss_real = self.adv_loss_fn(logits_real, True, True)
                total_d_loss = 0.5 * (d_loss_fake + d_loss_real)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * total_d_loss

            log = {
                f"{split}/disc_loss": d_loss.clone().detach().cpu().mean(),
                f"{split}/d_loss_fake": d_loss_fake.detach().cpu().mean(),
                f"{split}/d_loss_real": d_loss_real.detach().cpu().mean(),
            }
            return d_loss, log

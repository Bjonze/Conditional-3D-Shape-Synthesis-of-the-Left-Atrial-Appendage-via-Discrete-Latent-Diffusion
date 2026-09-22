"""Loading a trained diffusion model and sampling shapes from descriptors."""

import numpy as np
import torch

from laa_ldm.utils.config import instantiate_from_config, load_yaml_config
from laa_ldm.utils.mesh import largest_connected_component
from laa_ldm.utils.misc import get_model_parameters_info

__all__ = ["LatentDiffusionSampler"]


class LatentDiffusionSampler:
    """Generate LAA volumes for given shape descriptors.

    Args:
        config: path to the diffusion config the model was trained with.
        checkpoint: path to a training checkpoint (``last.pth``, ``best_*.pth``).
        use_ema: load the EMA weights, which is what the paper reports.
        device: device to run on.
    """

    def __init__(self, config, checkpoint, use_ema=True, device=None, verbose=True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.config = load_yaml_config(config)

        model = instantiate_from_config(self.config["model"])
        state = torch.load(checkpoint, map_location="cpu")

        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        if verbose:
            print(get_model_parameters_info(model))
            print(f"loaded {checkpoint}: {len(missing)} missing, {len(unexpected)} unexpected keys")

        if use_ema and "ema" in state:
            model.get_ema_model().load_state_dict(state["ema"], strict=False)
            if verbose:
                print("using EMA weights")

        self.epoch = state.get("last_epoch", state.get("epoch", 0))
        self.model = model.to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def configure_sampling(self, guidance_scale=5.0, learnable_cf=True, prior_rule=2,
                           prior_weight=1.0):
        """Set the classifier-free guidance and token-revealing behaviour.

        Args:
            guidance_scale: classifier-free guidance strength; 1.0 disables it.
            learnable_cf: use the learned empty embedding as the unconditional
                branch (as trained), rather than a zero descriptor vector.
            prior_rule: 0 = plain VQ-Diffusion, 1 = high-quality inference,
                2 = purity prior (reveal the most confident tokens first).
            prior_weight: strength of the purity prior ('r' in the Improved
                VQ-Diffusion paper); only used when ``prior_rule == 2``.
        """
        self.model.guidance_scale = guidance_scale
        self.model.learnable_cf = self.model.transformer.learnable_cf = learnable_cf
        self.model.transformer.prior_rule = prior_rule
        self.model.transformer.prior_weight = prior_weight

    @torch.no_grad()
    def sample(self, descriptors, replicate=1, truncation_rate=1.0, infer_speed=None,
               binarise=True, keep_largest_component=True):
        """Sample occupancy volumes for a batch of descriptor vectors.

        Args:
            descriptors: tensor or array of shape ``(B, 18)`` or ``(B, 18, 1)``.
            replicate: number of samples per descriptor vector.
            truncation_rate: nucleus truncation r of the reverse step.
            infer_speed: if set, skip steps to sample this many times faster
                (e.g. 2 for 2x); ``None`` runs all diffusion steps.
            binarise: threshold the decoded occupancy at 0.5.
            keep_largest_component: drop disconnected specks (requires
                ``binarise``).

        Returns:
            ``np.ndarray`` of shape ``(B * replicate, D, H, W)``.
        """
        if not torch.is_tensor(descriptors):
            descriptors = torch.as_tensor(np.asarray(descriptors), dtype=torch.float32)
        descriptors = descriptors.float()
        if descriptors.dim() == 2:
            descriptors = descriptors.unsqueeze(2)  # (B, 18, 1)
        descriptors = descriptors.to(self.device)

        sample_type = "top{}r".format(truncation_rate)
        if infer_speed is not None:
            sample_type += ",time{}".format(infer_speed)

        out = self.model.generate_content(
            batch={"ctx": descriptors, "indices": None},
            filter_ratio=0,
            replicate=replicate,
            content_ratio=1,
            sample_type=sample_type,
        )

        volumes = out["content"][:, 0].clamp(min=0.0, max=1.0).detach().cpu().numpy()
        if binarise:
            volumes = (volumes >= 0.5).astype(np.float32)
            if keep_largest_component:
                volumes = np.stack([largest_connected_component(v) for v in volumes])
        return volumes

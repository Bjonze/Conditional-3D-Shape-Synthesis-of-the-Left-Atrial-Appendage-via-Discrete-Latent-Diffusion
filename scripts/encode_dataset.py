"""Export VQ-GAN token grids for the diffusion stage.

The bridge between the two stages: runs the trained, descriptor-conditioned
encoder over every case and writes one ``.npz`` per LAA holding

* ``indices``: the flattened 8x8x8 grid of codebook indices (512 int32), and
* ``ctx``: the 18 shape descriptors used to condition the encoder.

    python scripts/encode_dataset.py --config configs/vqgan3d.yaml \\
        --checkpoint /path/to/vqgan3d.ckpt --output /path/to/latent_codes
"""

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from laa_ldm.codec.inference import load_vqgan
from laa_ldm.codec.training import build_datamodule
from laa_ldm.utils.config import load_yaml_config


@torch.no_grad()
def export_split(model, dataset, out_dir, device):
    """Encode every sample of ``dataset`` and write one npz per case."""
    os.makedirs(out_dir, exist_ok=True)
    for i in tqdm(range(len(dataset)), desc=f"encoding {os.path.basename(out_dir)}"):
        mask, context, file_name = dataset[i]
        mask = mask.unsqueeze(0).to(device)
        context = context.unsqueeze(0).to(device)

        quant, _, (_, _, indices) = model.encode(mask, context)
        _, _, d, h, w = quant.shape
        indices = indices.view(1, d * h * w)

        np.savez(
            os.path.join(out_dir, file_name),
            indices=indices[0].cpu().numpy().astype("int32"),
            ctx=context[0].cpu().numpy().astype("float32"),
        )


def main():
    parser = argparse.ArgumentParser(description="Export VQ-GAN latent codes.")
    parser.add_argument("--config", type=str, default="configs/vqgan3d.yaml",
                        help="VQ-GAN config; its paths define the dataset splits.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Trained VQ-GAN checkpoint.")
    parser.add_argument("--output", type=str, required=True,
                        help="Directory to write train/ and val/ token files into.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_yaml_config(args.config)
    dm = build_datamodule(cfg, with_name=True)

    model = load_vqgan(args.config, args.checkpoint).to(device).eval()

    export_split(model, dm.train_ds, os.path.join(args.output, "train"), device)
    export_split(model, dm.val_ds, os.path.join(args.output, "val"), device)
    print(f"Wrote latent codes to {args.output}")


if __name__ == "__main__":
    main()

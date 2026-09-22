"""Generate shapes conditioned on the descriptors of a dataset split.

Runs the trained diffusion model over every case of a token split (typically
``val``) using that case's real descriptors, so generated and reference shapes
can be compared one to one.

    python scripts/generate_from_descriptors.py \\
        --config configs/diffusion.yaml \\
        --checkpoint outputs/laa_diffusion/checkpoint/last.pth \\
        --data-root /path/to/latent_codes --output results/validation
"""

import argparse
import os

import torch
from tqdm import tqdm

from laa_ldm.diffusion.data import LatentTokenDataset
from laa_ldm.diffusion.inference import LatentDiffusionSampler
from laa_ldm.utils.mesh import save_mesh, save_nifti
from laa_ldm.utils.misc import seed_everything


def get_args():
    parser = argparse.ArgumentParser(description="Generate LAA shapes from real descriptors.")
    parser.add_argument("--config", type=str, default="configs/diffusion.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True,
                        help="Directory of exported token files (holds train/ and val/).")
    parser.add_argument("--phase", type=str, default="val", help="Split to generate for.")
    parser.add_argument("--output", type=str, required=True, help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=4)

    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--prior-rule", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--truncation-rate", type=float, default=1.0)
    parser.add_argument("--infer-speed", type=float, default=None,
                        help="Skip diffusion steps to sample this many times faster.")
    parser.add_argument("--no-learnable-cf", action="store_true",
                        help="Use a zero descriptor vector as the unconditional branch.")

    parser.add_argument("--save-volumes", action="store_true", default=True,
                        help="Write the occupancy volumes as .nii.gz.")
    parser.add_argument("--save-meshes", action="store_true",
                        help="Also write marching-cubes surfaces as .stl.")
    parser.add_argument("--keep-all-components", action="store_true",
                        help="Do not drop disconnected components.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = get_args()
    seed_everything(args.seed)

    os.makedirs(args.output, exist_ok=True)
    mesh_dir = os.path.join(args.output, "meshes")
    volume_dir = os.path.join(args.output, "volumes")
    if args.save_meshes:
        os.makedirs(mesh_dir, exist_ok=True)
    if args.save_volumes:
        os.makedirs(volume_dir, exist_ok=True)

    dataset = LatentTokenDataset(args.data_root, args.phase, with_name=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    sampler = LatentDiffusionSampler(args.config, args.checkpoint)
    sampler.configure_sampling(
        guidance_scale=args.guidance_scale,
        learnable_cf=not args.no_learnable_cf,
        prior_rule=args.prior_rule,
        prior_weight=args.prior_weight,
    )

    for batch in tqdm(loader, desc=f"generating {args.phase}"):
        volumes = sampler.sample(
            batch["ctx"],
            truncation_rate=args.truncation_rate,
            infer_speed=args.infer_speed,
            keep_largest_component=not args.keep_all_components,
        )
        for name, volume in zip(batch["name"], volumes):
            if args.save_volumes:
                save_nifti(volume, os.path.join(volume_dir, f"{name}.nii.gz"))
            if args.save_meshes:
                save_mesh(volume, os.path.join(mesh_dir, f"{name}.stl"))

    print(f"Wrote {len(dataset)} samples to {args.output}")


if __name__ == "__main__":
    main()

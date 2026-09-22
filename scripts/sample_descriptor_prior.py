"""Generate a synthetic cohort by sampling descriptors from a Gaussian prior.

The 18 descriptors are Box-Cox transformed and standardised, so their joint
distribution is approximately Gaussian.  Sampling that prior and conditioning
the diffusion model on the draws produces new anatomies that follow the
population statistics rather than copying any individual case.

The mean vector and covariance matrix of the training cohort ship in
``assets/statistics``.

    python scripts/sample_descriptor_prior.py \\
        --config configs/diffusion.yaml \\
        --checkpoint outputs/laa_diffusion/checkpoint/last.pth \\
        --num-samples 1000 --output results/prior_samples
"""

import argparse
import csv
import math
import os

import numpy as np
import pandas as pd

from laa_ldm.diffusion.inference import LatentDiffusionSampler
from laa_ldm.utils.mesh import save_mesh, save_nifti
from laa_ldm.utils.misc import seed_everything

STATS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "assets", "statistics")


def get_args():
    parser = argparse.ArgumentParser(description="Sample a synthetic LAA cohort.")
    parser.add_argument("--config", type=str, default="configs/diffusion.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)

    parser.add_argument("--covariance", type=str,
                        default=os.path.join(STATS_DIR, "covariance_matrix.csv"))
    parser.add_argument("--mean", type=str, default=os.path.join(STATS_DIR, "mean_vector.csv"),
                        help="Mean descriptor vector; omit to sample around zero.")
    parser.add_argument("--quantiles", type=str, default=None,
                        help="CSV with Q_0.01/Q_0.99 columns to clip the draws to.")

    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--prior-rule", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--prior-weight", type=float, default=0.0)
    parser.add_argument("--truncation-rate", type=float, default=1.0)
    parser.add_argument("--infer-speed", type=float, default=None)
    parser.add_argument("--no-learnable-cf", action="store_true")

    parser.add_argument("--save-volumes", action="store_true",
                        help="Also write the occupancy volumes as .nii.gz.")
    parser.add_argument("--keep-all-components", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_covariance(path, eps=1e-6):
    """Load the descriptor covariance and project it to the nearest PSD matrix."""
    frame = pd.read_csv(path)
    if "Unnamed: 0" in frame.columns:
        frame = frame.drop(columns=["Unnamed: 0"])
    cov = frame.to_numpy(dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"Covariance must be square; got {cov.shape}")

    cov = 0.5 * (cov + cov.T)
    # Clip tiny negative eigenvalues so the sampler cannot fail on round-off.
    w, v = np.linalg.eigh(cov)
    cov = (v * np.maximum(w, eps)) @ v.T
    return 0.5 * (cov + cov.T)


def load_mean(path, dim):
    arr = pd.read_csv(path).to_numpy(dtype=np.float32).squeeze()
    if arr.ndim != 1 or arr.shape[0] != dim:
        raise ValueError(f"Mean must have length {dim}; got {arr.shape}")
    return arr


def load_quantile_clip(path, dim):
    frame = pd.read_csv(path)
    if not {"Q_0.01", "Q_0.99"}.issubset(frame.columns):
        raise ValueError("Quantile file needs Q_0.01 and Q_0.99 columns.")
    q01 = frame["Q_0.01"].to_numpy(dtype=np.float32)
    q99 = frame["Q_0.99"].to_numpy(dtype=np.float32)
    if q01.shape[0] != dim:
        raise ValueError(f"Quantile dim {q01.shape[0]} != descriptor dim {dim}")
    return q01, q99


def main():
    args = get_args()
    seed_everything(args.seed)

    os.makedirs(args.output, exist_ok=True)
    mesh_dir = os.path.join(args.output, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    volume_dir = os.path.join(args.output, "volumes")
    if args.save_volumes:
        os.makedirs(volume_dir, exist_ok=True)

    cov = load_covariance(args.covariance)
    dim = cov.shape[0]
    mean = load_mean(args.mean, dim) if args.mean else np.zeros(dim, dtype=np.float32)

    rng = np.random.default_rng(args.seed)
    descriptors = rng.multivariate_normal(mean.astype(np.float64), cov,
                                          size=args.num_samples).astype(np.float32)
    if args.quantiles:
        q01, q99 = load_quantile_clip(args.quantiles, dim)
        descriptors = np.clip(descriptors, q01[None, :], q99[None, :])

    # Keep the descriptors so every mesh can be traced back to its conditioning.
    np.save(os.path.join(args.output, "descriptors.npy"), descriptors)

    sampler = LatentDiffusionSampler(args.config, args.checkpoint)
    sampler.configure_sampling(
        guidance_scale=args.guidance_scale,
        learnable_cf=not args.no_learnable_cf,
        prior_rule=args.prior_rule,
        prior_weight=args.prior_weight,
    )

    manifest_path = os.path.join(args.output, "manifest.csv")
    with open(manifest_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "mesh_file", "volume_file"])
        writer.writeheader()

        num_batches = math.ceil(args.num_samples / args.batch_size)
        for b in range(num_batches):
            start = b * args.batch_size
            stop = min(start + args.batch_size, args.num_samples)
            volumes = sampler.sample(
                descriptors[start:stop],
                truncation_rate=args.truncation_rate,
                infer_speed=args.infer_speed,
                keep_largest_component=not args.keep_all_components,
            )

            for i, volume in enumerate(volumes):
                sample_id = start + i
                mesh_file = os.path.join(mesh_dir, f"{sample_id:05d}.stl")
                save_mesh(volume, mesh_file)

                volume_file = ""
                if args.save_volumes:
                    volume_file = os.path.join(volume_dir, f"{sample_id:05d}.nii.gz")
                    save_nifti(volume, volume_file)

                writer.writerow({"sample_id": sample_id, "mesh_file": mesh_file,
                                 "volume_file": volume_file})

            if (b + 1) % 10 == 0 or (b + 1) == num_batches:
                print(f"Saved {stop}/{args.num_samples} samples...")

    print(f"Done. Meshes in {mesh_dir}, manifest at {manifest_path}")


if __name__ == "__main__":
    main()

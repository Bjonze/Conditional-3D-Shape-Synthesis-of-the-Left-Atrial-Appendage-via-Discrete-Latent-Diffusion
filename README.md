# Conditional 3D Shape Synthesis of the Left Atrial Appendage via Discrete Latent Diffusion

Code for the STACOM 2026 paper *"Conditional 3D Shape Synthesis of the Left Atrial Appendage via Discrete Latent Diffusion"*.

The left atrial appendage (LAA) is highly variable in shape, and that variability matters clinically — but cohorts of segmented appendages are small and hard to share. This repository generates new LAA anatomies **on demand and under anatomical control**: you specify 18 interpretable shape descriptors (volume, tortuosity, ostium axes, elongation, …) and the model synthesises a watertight 3D appendage that matches them.

The method has two stages:

```
                      18 shape descriptors
                              │
        ┌─────────────────────┼─────────────────────────┐
        │  FiLM               │  cross-attention        │
        ▼                     ▼                         │
┌───────────────┐      ┌──────────────┐                  │
│  Stage 1      │      │  Stage 2     │                  │
│  3D VQ-GAN    │      │  discrete    │                  │
│               │      │  diffusion   │                  │
│ 128³ mask     │      │              │                  │
│      ↓ encode │      │ 100 steps of │                  │
│  8×8×8 codes  │─────▶│ mask-and-    │                  │
│      ↓ decode │◀─────│ replace over │◀─────────────────┘
│ 128³ mask     │      │ 512 tokens   │
└───────────────┘      └──────────────┘
```

**Stage 1 — the codec.** A conditional 3D VQ-GAN compresses a 128³ binary LAA mask into an 8×8×8 grid of discrete codes drawn from a 512-entry codebook (512 tokens per shape). The descriptors modulate every residual block of the encoder through FiLM layers. The codebook is updated with EMA and cosine code matching, which keeps codebook usage high and avoids index collapse on binary volumes.

**Stage 2 — the prior.** A discrete diffusion model learns the distribution of those code grids. The forward process corrupts each token by either resampling it uniformly or replacing it with a `[MASK]` token; the reverse process is an 8-layer transformer that predicts `p(x₀ | x_t)`, attends to the descriptor tokens by cross-attention and receives the timestep through adaptive layer norm. Classifier-free guidance is trained by replacing the conditioning with a learned empty embedding 10% of the time.

Because the two stages are trained separately, the codec is frozen once trained: stage 2 never touches voxels, only the exported token grids.

## Repository layout

```
laa_ldm/
├── codec/                 Stage 1: the conditional 3D VQ-GAN
│   ├── blocks.py          FiLM, residual, attention, up/downsampling blocks
│   ├── nets.py            encoder and decoder cascades
│   ├── conditioning.py    descriptors → context tokens for FiLM
│   ├── quantize.py        EMA vector quantiser (cosine code matching)
│   ├── losses.py          weighted BCE + perceptual + patch-GAN loss
│   ├── model.py           the VQGAN3D LightningModule (training loop)
│   ├── inference.py       load a trained codec for encoding/decoding
│   ├── data.py            masks + descriptors dataset
│   └── training.py        trainer, callbacks and checkpointing
├── diffusion/             Stage 2: descriptor-conditioned discrete diffusion
│   ├── diffusion.py       the mask-and-replace diffusion process and its losses
│   ├── transformer.py     the denoising transformer (self + cross attention)
│   ├── embeddings.py      descriptor tokenizer and 3D token embedding
│   ├── condition.py       descriptor conditioning codec
│   ├── codec.py           frozen stage-1 decoder used to render samples
│   ├── model.py           the top-level model that ties the three together
│   ├── data.py            latent token dataset and dataloaders
│   ├── inference.py       sampling API for a trained model
│   └── engine/            solver, EMA, schedulers, gradient accumulation
├── utils/                 config, meshing, metrics, logging helpers
└── distributed.py         DDP helpers

configs/                   the two configurations used in the paper
scripts/                   command-line entry points
assets/statistics/         descriptor mean, covariance and quantiles
```

## Installation

```bash
conda create -n laa-ldm python=3.10
conda activate laa-ldm
pip install -r requirements.txt
pip install -e .
```

The models were trained with python 3.10, PyTorch 2.10 (CUDA 13.0), PyTorch Lightning 2.6, MONAI 1.5 and monai-generative 0.2.3. Weights & Biases is optional; logging is off in both configs.

## Data

Each case is a pair:

* a **binary LAA mask** as a 128³ `.nii.gz` volume, and
* its **18 shape descriptors**, standardised.

The descriptors live in a JSON file, one record per case:

```json
[{"filename": "case_0001.nii.gz", "tortuosity": -0.41, "centerline_length": 1.02, "volume": 0.77, "...": 0.0}]
```

The 18 keys, in the order the models expect them, are defined once in `laa_ldm/codec/data.py::DESCRIPTOR_KEYS`:

| | | |
|---|---|---|
| tortuosity | centerline_length | max_geodesic_distance |
| volume | angle_ostium_laa | cl_cut_25_elongation |
| cl_cut_25_cutarea | cl_cut_50_elongation | cl_cut_50_cutarea |
| cl_cut_75_elongation | cl_cut_75_cutarea | radii_95 |
| normalized_shape_index | elongation | flatness |
| surface_area | ostium_major_axis_length | ostium_minor_axis_length |

Point `configs/vqgan3d.yaml` at your mask directory and the two descriptor JSON files before training.

## Training

### Stage 1 — the VQ-GAN codec

```bash
python scripts/train_vqgan.py --config configs/vqgan3d.yaml
```

Multi-GPU is handled by Lightning through `trainer.devices` and `trainer.strategy` in the config. Gradient accumulation is done inside the module, so `trainer.gradient_accumulation.effective_batch_size` is the global batch size across all ranks. Validation writes reconstructions as `.stl` to `paths.image_dir`, and checkpoints are kept both on a rolling window and per best metric (IoU, precision).

### Export the latent codes

Stage 2 trains on token grids, not voxels, so encode the dataset once with the trained codec:

```bash
python scripts/encode_dataset.py \
    --config configs/vqgan3d.yaml \
    --checkpoint runs/vqgan3d/checkpoints/best_iou_epoch_101.ckpt \
    --output /path/to/latent_codes
```

This writes one `.npz` per case, holding the 512 codebook indices and the 18 descriptors, into `train/` and `val/` subdirectories.

### Stage 2 — the diffusion prior

Set `content_codec_config.params.ckpt_path` and the three `data_root` entries in `configs/diffusion.yaml`, then:

```bash
# all visible GPUs (one process per GPU)
python scripts/train_diffusion.py --config configs/diffusion.yaml

# a single GPU
python scripts/train_diffusion.py --config configs/diffusion.yaml --gpu 0

# resume
python scripts/train_diffusion.py --config configs/diffusion.yaml --auto_resume
```

Any config entry can be overridden from the command line as trailing `key.subkey value` pairs, e.g. `solver.max_epochs 400 dataloader.batch_size 4`.

Validation reports token accuracy overall and binned by diffusion timestep (`t0-33`, `t33-66`, `t66-99`), which separates "the model has learned the easy, nearly clean steps" from "the model has learned to denoise from a mostly masked grid". A checkpoint is kept for each of these metrics.

## Generation

### Conditioned on measured descriptors

Generate one shape per case of a split, using that case's real descriptors — this is the setting used to compare generated and reference anatomies one to one:

```bash
python scripts/generate_from_descriptors.py \
    --config configs/diffusion.yaml \
    --checkpoint outputs/laa_diffusion/checkpoint/last.pth \
    --data-root /path/to/latent_codes --phase val \
    --output results/validation --save-meshes
```

### A synthetic cohort from the descriptor prior

Because the descriptors are standardised, their joint distribution is approximately Gaussian. Sampling that prior produces new anatomies that follow the population statistics instead of copying any individual case. The training cohort's mean and covariance ship in `assets/statistics/`:

```bash
python scripts/sample_descriptor_prior.py \
    --config configs/diffusion.yaml \
    --checkpoint outputs/laa_diffusion/checkpoint/last.pth \
    --num-samples 1000 --output results/prior_samples
```

Each run writes the meshes, a `manifest.csv` and the `descriptors.npy` that produced them, so every sample can be traced back to its conditioning.

### From python

```python
from laa_ldm.diffusion.inference import LatentDiffusionSampler

sampler = LatentDiffusionSampler("configs/diffusion.yaml", "path/to/checkpoint.pth")
sampler.configure_sampling(guidance_scale=5.0, prior_rule=2, prior_weight=1.0)

volumes = sampler.sample(descriptors)          # (B, 18) -> (B, 128, 128, 128)
```

### Sampling knobs

| Flag | Default | Effect |
|---|---|---|
| `--guidance-scale` | 5.0 | Classifier-free guidance strength; 1.0 disables it. Higher values follow the descriptors more closely at some cost in diversity. |
| `--prior-rule` | 2 | 0 = plain VQ-Diffusion, 1 = high-quality inference, 2 = purity prior (reveal the most confident tokens first). |
| `--prior-weight` | 1.0 / 0.0 | Strength of the purity prior. Typical range: (0.0, 3.0) |
| `--truncation-rate` | 1.0 | Nucleus truncation of each reverse step. |
| `--infer-speed` | off | Skip diffusion steps to sample this many times faster. |

By default the decoded volume is thresholded at 0.5 and only the largest connected component is kept; pass `--keep-all-components` to disable that.

## Pretrained weights

*To be released.* Download links for the trained VQ-GAN and diffusion checkpoints will be added here.

| Stage | File | Notes |
|---|---|---|
| VQ-GAN codec | `vqgan3d.ckpt` | used as `content_codec_config.params.ckpt_path` |
| Diffusion prior | `diffusion.pth` | pass to `--checkpoint`; contains EMA weights |

## Citation

```bibtex
@article{laa_discrete_latent_diffusion,
  title   = {Conditional 3D Shape Synthesis of the Left Atrial Appendage via Discrete Latent Diffusion},
  author  = {Bj{\o}rn Hansen, Jonas Loft, Rasmus R. Paulsen, Oscar Camara, Klaus F. Kofoed and Kristine S{\o}rensen},
  journal = {Statistical Atlases and Computational Modeling of the Heart (STACOM), MICCAI Workshop},
  year    = {2026}
}
```

## Acknowledgements and licence

This code is released under the MIT License. It builds on [VQ-Diffusion](https://github.com/microsoft/VQ-Diffusion) (MIT), [MONAI GenerativeModels](https://github.com/Project-MONAI/GenerativeModels) (Apache-2.0) and [Taming Transformers](https://github.com/CompVis/taming-transformers) (MIT); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

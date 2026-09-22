"""Dataset and datamodule for the 3D VQ-GAN stage.

Each sample pairs a binary LAA mask (NIfTI, 128^3 voxels) with the 18 shape
descriptors of that appendage.  Descriptors are read from a JSON file produced
by the pre-processing pipeline: a list of records holding a ``filename`` plus
one entry per descriptor, already Box-Cox transformed and standardised.
"""

import json
import os

import SimpleITK as sitk
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

__all__ = ["DESCRIPTOR_KEYS", "LAAMaskDataset", "LAADataModule", "load_descriptor_json"]

#: The 18 conditioning descriptors, in the order the models expect them.
DESCRIPTOR_KEYS = [
    "tortuosity",
    "centerline_length",
    "max_geodesic_distance",
    "volume",
    "angle_ostium_laa",
    "cl_cut_25_elongation",
    "cl_cut_25_cutarea",
    "cl_cut_50_elongation",
    "cl_cut_50_cutarea",
    "cl_cut_75_elongation",
    "cl_cut_75_cutarea",
    "radii_95",
    "normalized_shape_index",
    "elongation",
    "flatness",
    "surface_area",
    "ostium_major_axis_length",
    "ostium_minor_axis_length",
]


def load_descriptor_json(path):
    """Load the list of descriptor records written by the pre-processing step."""
    with open(path, "r") as f:
        return json.load(f)


class LAAMaskDataset(torch.utils.data.Dataset):
    """Binary LAA masks paired with their 18 descriptors.

    Args:
        data_dir: directory holding the ``.nii.gz`` masks.
        json_path: descriptor JSON for this split.
        with_name: also return the file name (used when exporting latents).
        bad_filenames_path: optional text file listing cases to exclude, one
            stem per line (as ``<name>_labels``).
    """

    def __init__(self, data_dir, json_path, with_name=False, bad_filenames_path=None):
        super().__init__()
        self.data_dir = data_dir
        self.records = load_descriptor_json(json_path)
        self.keys = DESCRIPTOR_KEYS
        self.with_name = with_name

        if bad_filenames_path is not None and bad_filenames_path != "None":
            with open(bad_filenames_path, "r") as f:
                bad = {line.strip() for line in f if line.strip()}
            before = len(self.records)
            self.records = [
                r for r in self.records
                if r["filename"].replace(".nii.gz", "_labels") not in bad
            ]
            print(f"[LAAMaskDataset] Filtered {before - len(self.records)} bad entries "
                  f"using {bad_filenames_path} (kept {len(self.records)}/{before}).")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        file_name = record["filename"]

        mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_dir, file_name)))
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # (1, D, H, W)

        context = torch.tensor([record[key] for key in self.keys], dtype=torch.float32)
        context = context.unsqueeze(1)  # (18, 1)

        if self.with_name:
            return mask, context, file_name
        return mask, context


class LAADataModule(pl.LightningDataModule):
    """Train/validation loaders over :class:`LAAMaskDataset`."""

    def __init__(self, data_dir, json_path_train, json_path_val, batch_size=1, num_workers=8,
                 with_name=False, bad_filenames_path=None):
        super().__init__()
        self.data_dir = data_dir
        self.json_path_train = json_path_train
        self.json_path_val = json_path_val
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.with_name = with_name
        self.bad_filenames_path = bad_filenames_path

    def setup(self, stage=None):
        self.train_ds = LAAMaskDataset(self.data_dir, self.json_path_train,
                                       with_name=self.with_name,
                                       bad_filenames_path=self.bad_filenames_path)
        self.val_ds = LAAMaskDataset(self.data_dir, self.json_path_val,
                                     with_name=self.with_name,
                                     bad_filenames_path=self.bad_filenames_path)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=False)

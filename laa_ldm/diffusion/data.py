"""Dataset and dataloaders for the latent diffusion stage.

Stage 2 never touches voxels: it trains on the token grids exported once by
``scripts/encode_dataset.py``.  Each ``.npz`` holds the flattened codebook
indices of one LAA plus its 18 descriptors.
"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from laa_ldm.utils.config import instantiate_from_config

__all__ = ["LatentTokenDataset", "build_dataloader"]


class LatentTokenDataset(Dataset):
    """VQ-GAN token grids paired with their shape descriptors.

    Args:
        data_root: directory holding the ``train``/``val`` subfolders.
        phase: which subfolder to read.
        max_len: expected number of tokens per sample (8*8*8 = 512).
        with_name: also return the case name, for traceable exports.
    """

    def __init__(self, data_root, phase, max_len=512, dtype_idx=np.int64,
                 dtype_ctx=np.float32, with_name=False):
        self.paths = sorted(glob.glob(os.path.join(data_root, phase, "*.npz")))
        assert len(self.paths) > 0, f"No npz files found in {data_root}/{phase}"
        self.max_len = max_len
        self.dtype_idx = dtype_idx
        self.dtype_ctx = dtype_ctx
        self.with_name = with_name

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        arr = np.load(self.paths[i])
        idx = arr["indices"].astype(self.dtype_idx).reshape(-1)
        assert idx.shape[0] == self.max_len, f"Expected {self.max_len} tokens, got {idx.shape}"
        ctx = arr["ctx"].astype(self.dtype_ctx).reshape(-1)  # (18,)

        sample = {
            "indices": torch.from_numpy(idx),       # (512,)
            "ctx": torch.from_numpy(ctx).float(),   # (18,)
        }
        if self.with_name:
            name = os.path.basename(self.paths[i])
            name = name[:name.rfind(".nii.gz.npz")] if ".nii.gz.npz" in name else name[:-len(".npz")]
            sample["name"] = name
        return sample


def build_dataloader(config, args=None, return_dataset=False):
    """Build the train/validation loaders described by the ``dataloader`` block.

    Returns a dict with the loaders and their iteration counts; a
    ``DistributedSampler`` is used when ``args.distributed`` is set.
    """
    dataset_cfg = config['dataloader']

    def _build(split_key):
        datasets = []
        for ds_cfg in dataset_cfg[split_key]:
            ds_cfg['params']['data_root'] = dataset_cfg.get('data_root', '')
            datasets.append(instantiate_from_config(ds_cfg))
        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    train_dataset = _build('train_datasets')
    val_dataset = _build('validation_datasets')

    if args is not None and args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        train_iters = len(train_sampler) // dataset_cfg['batch_size']
        val_iters = len(val_sampler) // dataset_cfg['batch_size']
    else:
        train_sampler = None
        val_sampler = None
        train_iters = len(train_dataset) // dataset_cfg['batch_size']
        val_iters = len(val_dataset) // dataset_cfg['batch_size']

    num_workers = dataset_cfg['num_workers']
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=dataset_cfg['batch_size'],
                                               shuffle=(train_sampler is None),
                                               num_workers=num_workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last=True,
                                               persistent_workers=num_workers > 0)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=dataset_cfg['batch_size'],
                                             shuffle=False,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             sampler=val_sampler,
                                             drop_last=True,
                                             persistent_workers=num_workers > 0)

    dataload_info = {
        'train_loader': train_loader,
        'validation_loader': val_loader,
        'train_iterations': train_iters,
        'validation_iterations': val_iters,
    }
    if return_dataset:
        dataload_info['train_dataset'] = train_dataset
        dataload_info['validation_dataset'] = val_dataset
    return dataload_info

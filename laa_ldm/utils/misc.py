"""Small helpers: seeding, parameter counting, time formatting."""

import random
import warnings

import numpy as np
import torch

__all__ = ["seed_everything", "get_model_parameters_info", "format_seconds"]


def seed_everything(seed, cudnn_deterministic=False):
    """Seed python, numpy and torch."""
    if seed is not None:
        print(f"Global seed set to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This turns on the CUDNN deterministic "
            "setting, which can slow down training considerably."
        )


def get_model_parameters_info(model):
    """Trainable / non-trainable parameter counts per direct child module."""
    parameters = {"overall": {"trainable": 0, "non_trainable": 0, "total": 0}}
    for child_name, child_module in model.named_children():
        parameters[child_name] = {"trainable": 0, "non_trainable": 0}
        for _, p in child_module.named_parameters():
            key = "trainable" if p.requires_grad else "non_trainable"
            parameters[child_name][key] += p.numel()
        parameters[child_name]["total"] = (
            parameters[child_name]["trainable"] + parameters[child_name]["non_trainable"]
        )
        parameters["overall"]["trainable"] += parameters[child_name]["trainable"]
        parameters["overall"]["non_trainable"] += parameters[child_name]["non_trainable"]
        parameters["overall"]["total"] += parameters[child_name]["total"]

    def format_number(num):
        for unit, size in (("G", 2**30), ("M", 2**20), ("K", 2**10)):
            if num > size:
                return "{}{}".format(round(float(num) / size, 2), unit)
        return "{}".format(num)

    def format_dict(d):
        for k, v in d.items():
            if isinstance(v, dict):
                format_dict(v)
            else:
                d[k] = format_number(v)

    format_dict(parameters)
    return parameters


def format_seconds(seconds):
    """Format a duration as ``[Dd:]HHh:MMm:SSs``."""
    h = int(seconds // 3600)
    m = int(seconds // 60 - h * 60)
    s = int(seconds % 60)
    d = int(h // 24)
    h = h - d * 24

    if d > 0:
        return "{:d}d:{:02d}h:{:02d}m:{:02d}s".format(d, h, m, s)
    if h > 0:
        return "{:02d}h:{:02d}m:{:02d}s".format(h, m, s)
    if m > 0:
        return "{:02d}m:{:02d}s".format(m, s)
    return "{:02d}s".format(s)

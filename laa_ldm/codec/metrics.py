"""Voxel overlap metrics and the learning-rate schedule used by the VQ-GAN."""

import math

__all__ = ["compute_metrics", "CosineAnnealingWarmup"]


def compute_metrics(pred, target):
    """Precision, recall, F1 and IoU between a probability map and a binary target.

    ``pred`` is thresholded at 0.5; both tensors are compared over the whole batch.
    """
    epsilon = 1e-8
    pred = (pred > 0.5).float()
    target = target.float()

    TP = (pred * target).sum().item()
    FP = (pred * (1 - target)).sum().item()
    FN = ((1 - pred) * target).sum().item()

    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    iou = TP / (TP + FP + FN + epsilon)

    return precision, recall, f1, iou


class CosineAnnealingWarmup:
    """LambdaLR factor: linear warmup followed by cosine decay to zero.

    Args:
        num_steps: total number of optimiser steps.
        num_warmup_steps: steps spent ramping the factor from 0 to 1.
    """

    def __init__(self, num_steps, num_warmup_steps):
        self.num_steps = num_steps
        self.num_warmup_steps = num_warmup_steps

    def __call__(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        progress = float(current_step - self.num_warmup_steps) / float(
            max(1, self.num_steps - self.num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

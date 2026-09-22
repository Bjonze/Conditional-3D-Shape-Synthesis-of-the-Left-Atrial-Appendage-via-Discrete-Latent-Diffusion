"""Gradient accumulation, so a small per-GPU batch reaches the target batch size."""

from typing import Callable, Iterable, Optional

from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.optimizer import Optimizer

__all__ = ["GradientAccumulation"]


class GradientAccumulation:
    """
    Implements gradient accumulation to facilitate training with larger 
    effective batch sizes than what can be physically accommodated in memory.
    """

    def __init__(self,
                 actual_batch_size: int, 
                 expect_batch_size: int,
                 loader_len: int,
                 optimizer: Optimizer, 
                 grad_scaler: Optional[GradScaler] = None,
                 clip_grad_norm_fn: Optional[Callable[[Iterable], None]] = None,
                 clip_params: Optional[Iterable] = None) -> None:
        """
        Initializes the GradientAccumulation instance with the necessary parameters for 
        managing gradient accumulation.

        Args:
            actual_batch_size (int): The size of the mini-batches actually used in training.
            expect_batch_size (int): The desired (effective) batch size to simulate through gradient accumulation.
            loader_len (int): The length of the data loader, representing the total number of mini-batches.
            optimizer (Optimizer): The optimizer used for performing optimization steps.
            grad_scaler (Optional[GradScaler], optional): A GradScaler for mixed precision training. Defaults to None.
        
        Raises:
            AssertionError: If `expect_batch_size` is not divisible by `actual_batch_size`.
        """

        assert expect_batch_size % actual_batch_size == 0, \
            'expect_batch_size must be divisible by actual_batch_size'
        self.actual_batch_size = actual_batch_size
        self.expect_batch_size = expect_batch_size
        self.loader_len = loader_len
        self.optimizer = optimizer
        self.grad_scaler = grad_scaler
        self.clip_grad_norm_fn = clip_grad_norm_fn
        self.clip_params = clip_params

        # if the expected batch size is N=KM, and the actual batch size
        # is M, then we need to accumulate gradient from N / M = K optimization steps. 
        self.steps_until_update = expect_batch_size / actual_batch_size

    def step(self, loss: Tensor, step: int) -> bool:
        """
        Backward on scaled loss; when enough steps have accumulated,
        (optionally) clip gradients, optimizer.step(), and zero_grad().

        Returns:
            bool: True if an optimizer step was taken on this call, else False.
        """
        # Scale for accumulation
        loss = loss / self.steps_until_update
        
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()        

        took_step = False
        do_update = ((step + 1) % self.steps_until_update == 0) or ((step + 1) == self.loader_len)
        if do_update:
            # --- grad clipping happens here, before optimizer.step ---
            if self.clip_grad_norm_fn is not None and self.clip_params is not None:
                if self.grad_scaler is not None:
                    # unscale grads first when using AMP
                    self.grad_scaler.unscale_(self.optimizer)
                self.clip_grad_norm_fn(self.clip_params)

            # optimizer step
            if self.grad_scaler is not None:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)
            took_step = True

        return took_step

"""Distributed helpers for the diffusion stage.

Thin wrappers around ``torch.distributed`` that degrade to no-ops when training
on a single process, plus the spawn helper used by
``scripts/train_diffusion.py``.

Adapted from VQ-Diffusion (Microsoft, MIT licence).
"""

import socket

import torch
from torch import distributed as dist
from torch import multiprocessing as mp

__all__ = [
    "is_primary",
    "get_rank",
    "get_world_size",
    "synchronize",
    "all_reduce",
    "reduce_dict",
    "launch",
]

LOCAL_PROCESS_GROUP = None


def _dist_ready():
    return dist.is_available() and dist.is_initialized()


def is_primary():
    """True on rank 0 (or when not running distributed)."""
    return get_rank() == 0


def get_rank():
    return dist.get_rank() if _dist_ready() else 0


def get_local_rank():
    if not _dist_ready():
        return 0
    if LOCAL_PROCESS_GROUP is None:
        raise ValueError("LOCAL_PROCESS_GROUP is None")
    return dist.get_rank(group=LOCAL_PROCESS_GROUP)


def get_world_size():
    return dist.get_world_size() if _dist_ready() else 1


def synchronize():
    """Barrier across all ranks."""
    if _dist_ready() and dist.get_world_size() > 1:
        dist.barrier()


def all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False):
    if get_world_size() == 1:
        return tensor
    dist.all_reduce(tensor, op=op, async_op=async_op)
    return tensor


def reduce_dict(input_dict, average=True):
    """Reduce a dict of scalar tensors onto rank 0, averaging by default."""
    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        keys = sorted(input_dict.keys())
        values = torch.stack([input_dict[k] for k in keys], 0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            values /= world_size
        return {k: v for k, v in zip(keys, values)}


def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def launch(fn, n_gpu_per_machine, n_machine=1, machine_rank=0, dist_url=None, args=()):
    """Run ``fn(local_rank, *args)``, spawning one process per GPU if needed."""
    world_size = n_machine * n_gpu_per_machine

    if world_size <= 1:
        fn(0, *args)
        return

    if dist_url == "auto":
        if n_machine != 1:
            raise ValueError('dist_url="auto" is not supported in multi-machine jobs')
        dist_url = f"tcp://127.0.0.1:{find_free_port()}"

    if n_machine > 1 and dist_url.startswith("file://"):
        raise ValueError("file:// is not a reliable init method in multi-machine jobs; use tcp://")

    mp.spawn(
        _distributed_worker,
        nprocs=n_gpu_per_machine,
        args=(fn, world_size, n_gpu_per_machine, machine_rank, dist_url, args),
        daemon=False,
    )


def _distributed_worker(local_rank, fn, world_size, n_gpu_per_machine, machine_rank, dist_url,
                        args):
    global LOCAL_PROCESS_GROUP

    if not torch.cuda.is_available():
        raise OSError("CUDA is not available. Please check your environment.")
    if n_gpu_per_machine > torch.cuda.device_count():
        raise ValueError(f"Requested {n_gpu_per_machine} GPUs but only "
                         f"{torch.cuda.device_count()} are available")

    global_rank = machine_rank * n_gpu_per_machine + local_rank
    try:
        dist.init_process_group(backend="NCCL", init_method=dist_url, world_size=world_size,
                                rank=global_rank)
    except Exception as exc:
        raise OSError("failed to initialize NCCL groups") from exc

    synchronize()
    torch.cuda.set_device(local_rank)

    if LOCAL_PROCESS_GROUP is not None:
        raise ValueError("LOCAL_PROCESS_GROUP is already set")

    for i in range(world_size // n_gpu_per_machine):
        ranks_on_i = list(range(i * n_gpu_per_machine, (i + 1) * n_gpu_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            LOCAL_PROCESS_GROUP = pg

    fn(local_rank, *args)

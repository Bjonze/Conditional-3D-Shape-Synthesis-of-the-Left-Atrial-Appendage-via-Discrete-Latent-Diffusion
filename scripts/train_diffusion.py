"""Train the discrete latent diffusion model (stage 2).

Single GPU:

    python scripts/train_diffusion.py --config configs/diffusion.yaml --gpu 0

All visible GPUs (one process each, spawned by the script itself):

    python scripts/train_diffusion.py --config configs/diffusion.yaml

Adapted from VQ-Diffusion (Microsoft, MIT licence).
"""

import argparse
import os
import time
import warnings

import torch

from laa_ldm.diffusion.data import build_dataloader
from laa_ldm.diffusion.engine.logger import Logger
from laa_ldm.diffusion.engine.solver import Solver
from laa_ldm.distributed import launch
from laa_ldm.utils.config import instantiate_from_config, load_yaml_config, merge_opts_to_config
from laa_ldm.utils.misc import seed_everything
from laa_ldm.utils.wandb_logging import wandb

NODE_RANK = int(os.environ.get('AZ_BATCHAI_TASK_INDEX', 0))
MASTER_ADDR, MASTER_PORT = os.environ.get('AZ_BATCH_MASTER_NODE', '127.0.0.1:29500').split(':')
DIST_URL = 'tcp://%s:%s' % (MASTER_ADDR, MASTER_PORT)

#: Applied before NCCL starts. The defaults disable peer-to-peer and
#: InfiniBand, which multi-GPU machines without NVLink between the cards need.
DDP_ENV = {
    "NCCL_DEBUG": "INFO",
    "NCCL_P2P_DISABLE": "1",
    "NCCL_IB_DISABLE": "1",
}


def get_args():
    parser = argparse.ArgumentParser(description='Train the LAA latent diffusion model.')
    parser.add_argument('--config', dest='config_file', type=str,
                        default='configs/diffusion.yaml', help='path of config file')
    parser.add_argument('--name', type=str, default='',
                        help='name of this run; defaults to the config file name')
    parser.add_argument('--output', type=str, default='outputs',
                        help='directory to save the results')
    parser.add_argument('--log_frequency', type=int, default=100,
                        help='logging frequency in iterations')
    parser.add_argument('--load_path', type=str, default=None,
                        help='checkpoint to load model weights from (no optimiser state)')
    parser.add_argument('--resume_name', type=str, default=None,
                        help='resume the run with this name')
    parser.add_argument('--auto_resume', action='store_true',
                        help='resume from last.pth in the run directory if present')

    # distributed training
    parser.add_argument('--num_node', type=int, default=1, help='number of nodes')
    parser.add_argument('--node_rank', type=int, default=NODE_RANK, help='rank of this node')
    parser.add_argument('--dist_url', type=str, default=DIST_URL,
                        help='url used to set up distributed training')
    parser.add_argument('--gpu', type=int, default=None,
                        help='train on this single GPU and disable DDP')
    parser.add_argument('--sync_bn', action='store_true', help='use synchronised batch norm')
    parser.add_argument('--tensorboard', action='store_true', help='also log to tensorboard')
    parser.add_argument('--timestamp', action='store_true', help='prefix the run name with a timestamp')

    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--cudnn_deterministic', action='store_true',
                        help='set cudnn.deterministic, at a cost in speed')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='automatic mixed precision')
    parser.add_argument("opts", help="Modify config options from the command line, as "
                                     "`key.subkey value` pairs", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    args.cwd = os.path.abspath(os.path.dirname(__file__))

    if args.resume_name is not None:
        args.name = args.resume_name
        args.config_file = os.path.join(args.output, args.resume_name, 'configs', 'config.yaml')
        args.auto_resume = True
    else:
        if args.name == '':
            args.name = os.path.basename(args.config_file).replace('.yaml', '')
        if args.timestamp:
            assert not args.auto_resume, "--timestamp makes the save directory hard to find again"
            args.name = time.strftime('%Y-%m-%d-%H-%M') + '-' + args.name

    args.save_dir = os.path.join(args.output, args.name)
    return args


def main():
    args = get_args()

    if args.seed is not None or args.cudnn_deterministic:
        seed_everything(args.seed, args.cudnn_deterministic)

    if args.gpu is not None:
        warnings.warn('A specific GPU was chosen; this disables DDP.')
        torch.cuda.set_device(args.gpu)
        args.ngpus_per_node = 1
        args.world_size = 1
    else:
        args.dist_url = "auto" if args.num_node == 1 else args.dist_url
        assert args.num_node >= 1
        args.ngpus_per_node = torch.cuda.device_count()
        args.world_size = args.ngpus_per_node * args.num_node

        for key, value in DDP_ENV.items():
            os.environ.setdefault(key, value)

    launch(main_worker, args.ngpus_per_node, args.num_node, args.node_rank, args.dist_url,
           args=(args,))


def main_worker(local_rank, args):
    args.local_rank = local_rank
    args.global_rank = args.local_rank + args.node_rank * args.ngpus_per_node
    args.distributed = args.world_size > 1

    config = merge_opts_to_config(load_yaml_config(args.config_file), args.opts)

    wandb_run = None
    if args.global_rank == 0 and config.get('wandb_bool', False):
        if wandb is None:
            raise ImportError("wandb_bool is set but wandb is not installed.")
        wandb_run = wandb.init(project="laa-ldm", name=config.get('name', args.name))

    logger = Logger(args)
    logger.save_config(config)

    model = instantiate_from_config(config['model'])
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    dataloader_info = build_dataloader(config, args)
    solver = Solver(config=config, args=args, model=model, dataloader=dataloader_info,
                    logger=logger)

    if args.load_path is not None:  # only load the model parameters
        solver.resume(path=args.load_path, load_optimizer_and_scheduler=False, load_others=False)
    if args.auto_resume:
        solver.resume()

    try:
        solver.train()
    finally:
        if wandb_run is not None:
            wandb.finish()


if __name__ == '__main__':
    main()

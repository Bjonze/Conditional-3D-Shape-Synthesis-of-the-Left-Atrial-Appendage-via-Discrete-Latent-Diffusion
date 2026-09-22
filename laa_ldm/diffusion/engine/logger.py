"""Run directory, text log and (optional) TensorBoard writer for a training run."""

import os
import time

import torch

from laa_ldm.distributed import is_primary
from laa_ldm.utils.config import save_config_to_yaml, write_args

__all__ = ["Logger"]


class Logger(object):
    """Owns ``<save_dir>/{configs,logs}`` and writes only from the primary rank."""

    def __init__(self, args):
        self.args = args
        self.save_dir = args.save_dir
        self.is_primary = is_primary()

        if self.is_primary:
            os.makedirs(self.save_dir, exist_ok=True)

            self.config_dir = os.path.join(self.save_dir, 'configs')
            os.makedirs(self.config_dir, exist_ok=True)
            write_args(args, os.path.join(self.config_dir, 'args.txt'))

            log_dir = os.path.join(self.save_dir, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            self.text_writer = open(os.path.join(log_dir, 'log.txt'), 'a')

            if getattr(args, 'tensorboard', False):
                self.log_info('using tensorboard')
                self.tb_writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir)
            else:
                self.tb_writer = None

    def save_config(self, config):
        if self.is_primary:
            save_config_to_yaml(config, os.path.join(self.config_dir, 'config.yaml'))

    def log_info(self, info, check_primary=True):
        if self.is_primary or (not check_primary):
            print(info)
            if self.is_primary:
                info = '{}: {}'.format(time.strftime('%Y-%m-%d-%H-%M'), str(info))
                if not info.endswith('\n'):
                    info += '\n'
                self.text_writer.write(info)
                self.text_writer.flush()

    def add_scalar(self, **kargs):
        if self.is_primary and self.tb_writer is not None:
            self.tb_writer.add_scalar(**kargs)

    def add_scalars(self, **kargs):
        if self.is_primary and self.tb_writer is not None:
            self.tb_writer.add_scalars(**kargs)

    def close(self):
        if self.is_primary:
            self.text_writer.close()
            if self.tb_writer is not None:
                self.tb_writer.close()

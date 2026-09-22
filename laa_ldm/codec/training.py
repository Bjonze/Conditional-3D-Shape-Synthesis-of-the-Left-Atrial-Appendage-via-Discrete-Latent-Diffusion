"""Trainer assembly for the VQ-GAN stage.

Everything is driven by ``configs/vqgan3d.yaml``: the datamodule, the model,
the loggers, the checkpoint callbacks and the Lightning trainer.  See
``scripts/train_vqgan.py`` for the entry point.
"""

import os

import pytorch_lightning as pl

from laa_ldm.codec.data import LAADataModule
from laa_ldm.codec.model import VQGAN3D

__all__ = [
    "configure_distributed_environment",
    "build_datamodule",
    "build_model",
    "build_logger",
    "build_callbacks",
    "build_trainer",
    "RollingEpochCheckpoint",
]


def _requested_device_count(devices):
    if isinstance(devices, int):
        return devices
    if isinstance(devices, (list, tuple)):
        return len(devices)
    if isinstance(devices, str):
        value = devices.strip().lower()
        if value == "auto":
            return None
        if "," in value:
            return len([item for item in value.split(",") if item.strip()])
        try:
            return int(value)
        except ValueError:
            return None
    return None

def configure_distributed_environment(cfg: dict) -> None:
    """Set the NCCL environment variables required for multi-GPU training.

    Defaults disable P2P and InfiniBand, which is what multi-GPU boxes without
    NVLink between the cards need; override them under ``trainer.ddp_env``.
    """
    trainer_cfg = cfg.get("trainer", {})
    devices = trainer_cfg.get("devices", 1)
    strategy = str(trainer_cfg.get("strategy", "")).lower()
    device_count = _requested_device_count(devices)
    distributed_requested = "ddp" in strategy or device_count is None or device_count > 1

    if not distributed_requested:
        return

    ddp_env = {
        "NCCL_DEBUG": "INFO",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
    }
    ddp_env.update(trainer_cfg.get("ddp_env", {}) or {})

    for key, value in ddp_env.items():
        os.environ.setdefault(str(key), str(value))

def build_datamodule(cfg: dict, with_name: bool = False) -> LAADataModule:
    """Build and set up the mask/descriptor datamodule described by ``cfg``."""
    dm = LAADataModule(
        data_dir=cfg["paths"]["data_dir"],
        json_path_train=cfg["paths"]["descriptors"]["train"],
        json_path_val=cfg["paths"]["descriptors"]["val"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        with_name=with_name,
        bad_filenames_path=cfg["paths"]["descriptors"].get("bad_filenames"),
    )
    dm.setup()
    return dm

def build_model(cfg: dict, dataset_len: int) -> VQGAN3D:
    """Merge the model, optimiser and loss blocks of ``cfg`` into a VQGAN3D."""
    bs = cfg["data"]["batch_size"]
    ebs = cfg.get("trainer", {}).get("gradient_accumulation", {}).get("effective_batch_size", bs)

    # Base model (architecture) args
    margs = dict(cfg["model"])
    margs.update({
        "batch_size": bs,
        "E_batch_size": ebs,
        "dataset_len": dataset_len,
        "n_epochs": cfg["trainer"]["max_epochs"],
        "num_warmup_epochs": cfg["trainer"].get("num_warmup_epochs", 0),

        "lr_ae": cfg["optim"]["lr_ae"],
        "lr_disc": cfg["optim"]["lr_disc"],
    })

    margs["disc_cfg"] = cfg["discriminator"]
    margs["losses_cfg"] = cfg.get("losses", {})

    return VQGAN3D(**margs)

def build_logger(cfg: dict):
    """CSV logger, plus Weights & Biases when enabled in the config."""
    loggers = []
    logs_dir = cfg["paths"].get("logs_dir")
    if logs_dir:
        os.makedirs(logs_dir, exist_ok=True)
        loggers.append(pl.loggers.CSVLogger(save_dir=logs_dir, name=""))

    wcfg = cfg.get("logging", {}).get("wandb", {})
    if wcfg.get("enabled", False):
        loggers.append(pl.loggers.WandbLogger(
            project=wcfg.get("project", "VQGAN"),
            entity=wcfg.get("entity"),
            log_model=wcfg.get("log_model", False),
            name=wcfg.get("run_name"),
        ))

    if len(loggers) == 1:
        return loggers[0]
    return loggers

def build_trainer(cfg: dict, logger, callbacks) -> pl.Trainer:
    """Lightning trainer; grad accumulation is handled inside the module, not here."""
    trainer_cfg = cfg.get("trainer", {})
    devices = trainer_cfg.get("devices", 1)
    device_count = _requested_device_count(devices)

    trainer_kwargs = {
        "logger": logger,
        "accelerator": trainer_cfg.get("accelerator", "gpu"),
        "devices": devices,
        "precision": trainer_cfg.get("precision", 16),
        "max_epochs": trainer_cfg["max_epochs"],
        "callbacks": callbacks,
    }

    if "strategy" in trainer_cfg:
        trainer_kwargs["strategy"] = trainer_cfg["strategy"]
    elif device_count is None or device_count > 1:
        trainer_kwargs["strategy"] = "ddp"

    for key in ("num_nodes", "use_distributed_sampler", "sync_batchnorm"):
        if key in trainer_cfg:
            trainer_kwargs[key] = trainer_cfg[key]

    return pl.Trainer(**trainer_kwargs)

class RollingEpochCheckpoint(pl.Callback):
    """Save a checkpoint every ``every_n_epochs`` and keep only the newest ``keep_last``."""

    def __init__(
        self,
        dirpath: str,
        filename: str = "rolling_epoch_{epoch:03d}",
        keep_last: int = 5,
        every_n_epochs: int = 1,
        save_weights_only: bool = False,
    ) -> None:
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
        self.keep_last = int(keep_last)
        self.every_n_epochs = int(every_n_epochs)
        self.save_weights_only = bool(save_weights_only)

    def _checkpoint_path(self, epoch: int) -> str:
        filename = self.filename.format(epoch=epoch)
        if not filename.endswith(".ckpt"):
            filename = f"{filename}.ckpt"
        return os.path.join(self.dirpath, filename)

    def _remove_stale_checkpoints(self) -> None:
        if self.keep_last <= 0:
            return

        prefix = self.filename.split("{epoch", 1)[0]
        checkpoints = []
        for name in os.listdir(self.dirpath):
            if name.startswith(prefix) and name.endswith(".ckpt"):
                path = os.path.join(self.dirpath, name)
                checkpoints.append((os.path.getmtime(path), path))

        checkpoints.sort(reverse=True)
        for _, path in checkpoints[self.keep_last:]:
            os.remove(path)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        if self.every_n_epochs > 0 and (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        if trainer.is_global_zero:
            os.makedirs(self.dirpath, exist_ok=True)

        checkpoint_path = self._checkpoint_path(trainer.current_epoch)
        trainer.save_checkpoint(checkpoint_path, weights_only=self.save_weights_only)

        if trainer.is_global_zero:
            self._remove_stale_checkpoints()

        trainer.strategy.barrier()


def build_callbacks(cfg: dict):
    """A rolling last-k checkpoint plus best-precision and best-IoU checkpoints."""
    os.makedirs(cfg["paths"]["checkpoints_dir"], exist_ok=True)
    cb_cfg = cfg.get("callbacks", {}).get("model_checkpoint", {})

    rolling_cb = RollingEpochCheckpoint(
        dirpath=cfg["paths"]["checkpoints_dir"],
        filename=cb_cfg.get("rolling_filename", "last_epoch_{epoch:03d}"),
        keep_last=cb_cfg.get("rolling_last_k", 5),
        every_n_epochs=cb_cfg.get("every_n_epochs", 1),
        save_weights_only=cb_cfg.get("save_weights_only", False),
    )

    best_precision_cb = pl.callbacks.ModelCheckpoint(
        dirpath=cfg["paths"]["checkpoints_dir"],
        filename=cb_cfg.get("best_precision_filename", "best_precision_epoch_{epoch:03d}"),
        monitor=cb_cfg.get("precision_monitor", "val/precision"),
        mode="max",
        save_top_k=1,
        every_n_epochs=cb_cfg.get("every_n_epochs", 1),
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
    )

    best_iou_cb = pl.callbacks.ModelCheckpoint(
        dirpath=cfg["paths"]["checkpoints_dir"],
        filename=cb_cfg.get("best_iou_filename", "best_iou_epoch_{epoch:03d}"),
        monitor=cb_cfg.get("iou_monitor", "val/iou"),
        mode="max",
        save_top_k=1,
        every_n_epochs=cb_cfg.get("every_n_epochs", 1),
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
    )

    return [rolling_cb, best_precision_cb, best_iou_cb]

"""Train the 3D VQ-GAN (stage 1).

    python scripts/train_vqgan.py --config configs/vqgan3d.yaml
"""

import argparse
import os

from laa_ldm.codec.training import (
    build_callbacks,
    build_datamodule,
    build_logger,
    build_model,
    build_trainer,
    configure_distributed_environment,
)
from laa_ldm.utils.config import load_yaml_config


def main(cfg_path: str) -> None:
    cfg = load_yaml_config(cfg_path)
    configure_distributed_environment(cfg)

    dm = build_datamodule(cfg)
    model = build_model(cfg, dataset_len=len(dm.train_ds))

    # Where the module writes validation reconstructions and CSV logs.
    for key, attr in (("image_dir", "images_dir"), ("logs_dir", "logs_dir")):
        path = cfg["paths"].get(key)
        if path is not None:
            os.makedirs(path, exist_ok=True)
            setattr(model, attr, path)

    trainer = build_trainer(cfg, build_logger(cfg), build_callbacks(cfg))
    trainer.fit(model, dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the 3D VQ-GAN codec.")
    parser.add_argument("--config", type=str, default="configs/vqgan3d.yaml",
                        help="Path to the YAML config.")
    main(parser.parse_args().config)

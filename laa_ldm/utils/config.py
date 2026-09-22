"""YAML configuration helpers shared by both stages."""

import importlib
import sys

import torch
import yaml

__all__ = [
    "load_yaml_config",
    "save_config_to_yaml",
    "instantiate_from_config",
    "merge_opts_to_config",
    "write_args",
]


def load_yaml_config(path):
    with open(path) as f:
        return yaml.full_load(f)


def save_config_to_yaml(config, path):
    assert path.endswith(".yaml")
    with open(path, "w") as f:
        f.write(yaml.dump(config))


def instantiate_from_config(config):
    """Build the object described by ``{'target': 'pkg.mod.Cls', 'params': {...}}``."""
    if config is None:
        return None
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config["target"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    return cls(**config.get("params", dict()))


def merge_opts_to_config(config, opts):
    """Apply ``key.subkey value`` command-line overrides onto a loaded config."""

    def modify_dict(c, nl, v):
        if len(nl) == 1:
            c[nl[0]] = type(c[nl[0]])(v)
        else:
            c[nl[0]] = modify_dict(c[nl[0]], nl[1:], v)
        return c

    if opts is not None and len(opts) > 0:
        assert len(opts) % 2 == 0, "opts must be given as alternating names and values"
        for i in range(len(opts) // 2):
            config = modify_dict(config, opts[2 * i].split("."), opts[2 * i + 1])
    return config


def write_args(args, path):
    """Append the parsed CLI arguments and environment versions to ``path``."""
    args_dict = {name: getattr(args, name) for name in dir(args) if not name.startswith("_")}
    with open(path, "a") as args_file:
        args_file.write("==> torch version: {}\n".format(torch.__version__))
        args_file.write("==> cudnn version: {}\n".format(torch.backends.cudnn.version()))
        args_file.write("==> Cmd:\n")
        args_file.write(str(sys.argv))
        args_file.write("\n==> args:\n")
        for k, v in sorted(args_dict.items()):
            args_file.write("  %s: %s\n" % (str(k), str(v)))

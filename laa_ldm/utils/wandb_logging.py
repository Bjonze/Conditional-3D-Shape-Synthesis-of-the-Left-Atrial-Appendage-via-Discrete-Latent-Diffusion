"""Optional Weights & Biases logging.

``wandb`` is not a hard dependency: if it is missing, or if logging is disabled
in the config, every call here is a no-op.
"""

try:  # pragma: no cover - optional dependency
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

__all__ = ["wandb", "wandb_available", "wandb_log"]


def wandb_available():
    """True when wandb is installed and a run is active."""
    return wandb is not None and wandb.run is not None


def wandb_log(metrics, step=None):
    """Log a metrics dict to the active run, if there is one."""
    if wandb_available():
        wandb.log(metrics, step=step)

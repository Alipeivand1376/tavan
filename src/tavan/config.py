"""Configuration objects shared across neural models.

All neural models in ``tavan`` accept a :class:`TrainConfig` (or the same
fields as keyword arguments), so training behaviour is controlled in one
consistent place.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Dict, Optional


@dataclass
class TrainConfig:
    """Hyper-parameters of the optimisation loop.

    Attributes
    ----------
    epochs:
        Maximum number of passes over the training data.
    batch_size:
        Mini-batch size.
    lr:
        Initial learning rate.
    weight_decay:
        Decoupled (AdamW) weight decay.
    optimizer:
        ``"adam"``, ``"adamw"`` or ``"sgd"``.
    scheduler:
        ``"cosine"``, ``"step"`` or ``None``.
    warmup:
        Warm-up epochs for the cosine schedule.
    grad_clip:
        Global gradient-norm clipping threshold (``0`` disables).
    val_fraction:
        Fraction of samples held out for early stopping (``0`` disables).
    patience:
        Early-stopping patience in epochs (``0`` disables).
    min_delta:
        Minimum relative improvement counted by early stopping.
    seed:
        Random seed for initialisation, shuffling and sampling.
    verbose:
        ``0`` silent, ``1`` summary every ``log_every`` epochs.
    log_every:
        Logging frequency in epochs.
    """

    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    scheduler: Optional[str] = "cosine"
    warmup: int = 0
    grad_clip: float = 5.0
    val_fraction: float = 0.1
    patience: int = 30
    min_delta: float = 1e-4
    seed: Optional[int] = 0
    verbose: int = 0
    log_every: int = 25

    def update(self, **kwargs: Any) -> "TrainConfig":
        """Return a copy with selected fields replaced (unknown keys raise)."""
        valid = {f.name for f in fields(self)}
        bad = set(kwargs) - valid
        if bad:
            raise TypeError(f"unknown TrainConfig fields: {sorted(bad)}")
        return replace(self, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_any(cls, cfg: Optional["TrainConfig | Dict[str, Any]"] = None, **overrides: Any) -> "TrainConfig":
        if cfg is None:
            base = cls()
        elif isinstance(cfg, cls):
            base = cfg
        elif isinstance(cfg, dict):
            base = cls().update(**cfg)
        else:
            raise TypeError("train config must be a TrainConfig or dict")
        overrides = {k: v for k, v in overrides.items() if v is not None}
        return base.update(**overrides) if overrides else base


TRAIN_FIELDS = frozenset(f.name for f in fields(TrainConfig))


def split_train_kwargs(kwargs: Dict[str, Any]):
    """Split ``kwargs`` into (TrainConfig fields, remaining)."""
    train = {k: v for k, v in kwargs.items() if k in TRAIN_FIELDS}
    rest = {k: v for k, v in kwargs.items() if k not in TRAIN_FIELDS}
    return train, rest


__all__ = ["TrainConfig", "split_train_kwargs", "TRAIN_FIELDS", "field"]

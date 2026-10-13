"""Differentiable loss functions."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .autograd import Tensor


def _t(x) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


def mse(pred: Tensor, target, weight: Optional[np.ndarray] = None) -> Tensor:
    """Mean squared error."""
    d = pred - _t(target)
    sq = d * d
    if weight is not None:
        sq = sq * weight
    return sq.mean()


def mae(pred: Tensor, target) -> Tensor:
    """Mean absolute error."""
    return (pred - _t(target)).abs().mean()


def huber(pred: Tensor, target, delta: float = 1.0) -> Tensor:
    """Huber loss (smooth L1)."""
    d = (pred - _t(target)).abs()
    quad = d.clip(None, delta)
    lin = d - quad
    return (quad * quad * 0.5 + lin * delta).mean()


def pinball(pred: Tensor, target, quantiles: Sequence[float]) -> Tensor:
    """Quantile (pinball) loss.

    ``pred`` has a trailing axis of size ``len(quantiles)``.
    """
    q = np.asarray(quantiles, dtype=float)
    y = _t(target)
    if y.ndim == pred.ndim - 1:
        y = y.reshape(*y.shape, 1)
    diff = y - pred
    # max(q*d, (q-1)*d) == q*d - min(d, 0)  -> use relu formulation
    return (diff * q + (-diff).relu()).mean()


def gaussian_nll(mu: Tensor, log_var: Tensor, target) -> Tensor:
    """Gaussian negative log-likelihood (up to a constant)."""
    d = _t(target) - mu
    return (log_var + d * d / log_var.exp()).mean() * 0.5


def kl_standard_normal(mu: Tensor, log_var: Tensor) -> Tensor:
    """KL( N(mu, exp(log_var)) || N(0, I) ), averaged over the batch."""
    kl = (mu * mu + log_var.exp() - log_var - 1.0).sum(axis=-1) * 0.5
    return kl.mean()


def bce_with_logits(logits: Tensor, target) -> Tensor:
    """Numerically stable binary cross-entropy on logits."""
    y = _t(target)
    # softplus(x) - y*x
    return (logits.softplus() - logits * y).mean()


LOSSES = {"mse": mse, "mae": mae, "huber": huber}


def get_loss(name: str):
    try:
        return LOSSES[name]
    except KeyError:
        raise ValueError(f"unknown loss {name!r}; choose from {sorted(LOSSES)}") from None

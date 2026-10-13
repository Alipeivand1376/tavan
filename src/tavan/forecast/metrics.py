"""Forecast accuracy metrics (point and probabilistic)."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def _align(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return yt, yp


def mae(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def mse(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    """Mean absolute percentage error (%)."""
    yt, yp = _align(y_true, y_pred)
    return float(np.mean(np.abs((yt - yp) / (np.abs(yt) + eps))) * 100)


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    """Symmetric MAPE (%), bounded in [0, 200]."""
    yt, yp = _align(y_true, y_pred)
    return float(np.mean(2 * np.abs(yt - yp) / (np.abs(yt) + np.abs(yp) + eps)) * 100)


def mase(y_true, y_pred, y_train, season_length: int = 1) -> float:
    """Mean absolute scaled error (Hyndman & Koehler, 2006).

    Scaled by the in-sample seasonal-naive MAE; ``<1`` beats seasonal naive.
    """
    yt, yp = _align(y_true, y_pred)
    tr = np.asarray(y_train, dtype=float)
    scale = np.mean(np.abs(tr[season_length:] - tr[:-season_length])) + 1e-12
    return float(np.mean(np.abs(yt - yp)) / scale)


def r2(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2) + 1e-12
    return float(1 - ss_res / ss_tot)


def pinball_loss(y_true, y_quantiles: Dict[float, np.ndarray]) -> float:
    """Average pinball loss over the provided quantile levels."""
    yt = np.asarray(y_true, dtype=float)
    total = 0.0
    for q, pred in y_quantiles.items():
        d = yt - np.asarray(pred, dtype=float)
        total += np.mean(np.maximum(q * d, (q - 1) * d))
    return float(total / len(y_quantiles))


def quantile_coverage(y_true, y_quantiles: Dict[float, np.ndarray]) -> Dict[float, float]:
    """Empirical fraction of observations below each quantile forecast."""
    yt = np.asarray(y_true, dtype=float)
    return {q: float(np.mean(yt <= np.asarray(pred, dtype=float))) for q, pred in sorted(y_quantiles.items())}


POINT_METRICS = ("mae", "rmse", "mape", "smape", "r2")


def evaluate_point(y_true, y_pred, metrics: Sequence[str] = POINT_METRICS, y_train=None,
                   season_length: int = 1) -> Dict[str, float]:
    """Compute several point-forecast metrics at once."""
    funcs = {"mae": mae, "rmse": rmse, "mse": mse, "mape": mape, "smape": smape, "r2": r2,
             "mase": lambda a, b: mase(a, b, y_train, season_length)}
    out = {}
    for m in metrics:
        if m == "mase" and y_train is None:
            continue
        if m not in funcs:
            raise ValueError(f"unknown metric {m!r}; choose from {sorted(funcs)}")
        out[m] = funcs[m](y_true, y_pred)
    return out

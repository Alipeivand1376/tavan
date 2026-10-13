"""Outlier detection and cleaning."""

from __future__ import annotations

import numpy as np

from ..registry import register
from .base import BaseTransform


def _rolling_median_mad(col: np.ndarray, window: int):
    half = window // 2
    padded = np.pad(col, half, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(win, axis=1)
            mad = np.nanmedian(np.abs(win - med[:, None]), axis=1)
    return med, mad


@register("transform", "outliers", aliases=("outlier_cleaner",))
class OutlierCleaner(BaseTransform):
    """Detect outliers and replace them.

    Parameters
    ----------
    method:
        ``"hampel"`` (rolling median ± k·MAD, robust to level shifts),
        ``"zscore"`` (global mean ± k·std) or ``"iqr"`` (Tukey fences).
    threshold:
        ``k`` multiplier (Tukey uses ``threshold`` × IQR beyond quartiles).
    window:
        Window length for Hampel filtering.
    action:
        ``"nan"`` (mark as missing so an imputer can fill them),
        ``"clip"`` (clip to the fence) or ``"median"`` (replace with the
        rolling median).

    Attributes
    ----------
    mask_ : ndarray of bool
        Outlier positions found by the last ``transform``.
    """

    def __init__(self, method: str = "hampel", threshold: float = 3.5, window: int = 25,
                 action: str = "nan") -> None:
        self.method = method
        self.threshold = threshold
        self.window = window
        self.action = action

    def transform(self, x):
        arr = np.asarray(x, dtype=float)
        a2 = arr[:, None] if arr.ndim == 1 else arr.copy()
        out = a2.copy()
        mask = np.zeros_like(a2, dtype=bool)
        k = self.threshold
        for c in range(a2.shape[1]):
            col = a2[:, c]
            if self.method == "hampel":
                center, mad = _rolling_median_mad(col, self.window)
                spread = 1.4826 * mad
                spread = np.where(spread > 1e-12, spread, np.nanmedian(spread[spread > 1e-12]) if np.any(spread > 1e-12) else 1.0)
                lo, hi = center - k * spread, center + k * spread
            elif self.method == "zscore":
                mu, sd = np.nanmean(col), np.nanstd(col)
                center = np.full_like(col, mu)
                lo, hi = mu - k * sd, mu + k * sd
            elif self.method == "iqr":
                q1, q3 = np.nanpercentile(col, [25, 75])
                center = np.full_like(col, np.nanmedian(col))
                lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
            else:
                raise ValueError("method must be 'hampel', 'zscore' or 'iqr'")
            lo = np.broadcast_to(lo, col.shape)
            hi = np.broadcast_to(hi, col.shape)
            m = np.isfinite(col) & ((col < lo) | (col > hi))
            mask[:, c] = m
            if self.action == "nan":
                out[m, c] = np.nan
            elif self.action == "clip":
                out[:, c] = np.where(m, np.clip(col, lo, hi), col)
            elif self.action == "median":
                out[m, c] = np.broadcast_to(center, col.shape)[m]
            else:
                raise ValueError("action must be 'nan', 'clip' or 'median'")
        self.mask_ = mask[:, 0] if arr.ndim == 1 else mask
        return out[:, 0] if arr.ndim == 1 else out


def detect_outliers(x, method: str = "hampel", threshold: float = 3.5, window: int = 25) -> np.ndarray:
    """Boolean mask of outliers."""
    oc = OutlierCleaner(method=method, threshold=threshold, window=window)
    oc.transform(x)
    return oc.mask_

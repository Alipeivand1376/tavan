"""Reversible scalers (NaN-aware, per channel)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..registry import register
from .base import BaseTransform, _flatten_channels


class _ColumnScaler(BaseTransform):
    _state_attrs = ("center_", "scale_")

    def _stats(self, x2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        raise NotImplementedError

    def fit(self, x, y=None):
        x2, _ = _flatten_channels(x)
        with np.errstate(all="ignore"):
            c, s = self._stats(x2)
        s = np.where(np.isfinite(s) & (np.abs(s) > 1e-12), s, 1.0)
        self.center_ = np.nan_to_num(c)
        self.scale_ = s
        return self

    def transform(self, x):
        x2, restore = _flatten_channels(x)
        return restore((x2 - self.center_) / self.scale_)

    def inverse_transform(self, x):
        x2, restore = _flatten_channels(x)
        return restore(x2 * self.scale_ + self.center_)


@register("transform", "standard", aliases=("zscore", "standard_scaler"))
class StandardScaler(_ColumnScaler):
    """Zero mean, unit variance."""

    def __init__(self) -> None:
        pass

    def _stats(self, x2):
        return np.nanmean(x2, 0), np.nanstd(x2, 0)


@register("transform", "minmax", aliases=("minmax_scaler",))
class MinMaxScaler(_ColumnScaler):
    """Scale to ``feature_range`` (default ``[0, 1]``)."""

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)) -> None:
        self.feature_range = feature_range

    def _stats(self, x2):
        lo, hi = self.feature_range
        mn, mx = np.nanmin(x2, 0), np.nanmax(x2, 0)
        scale = (mx - mn) / (hi - lo)
        return mn - lo * scale, scale


@register("transform", "robust", aliases=("robust_scaler",))
class RobustScaler(_ColumnScaler):
    """Median / inter-quartile-range scaling, insensitive to outliers."""

    def __init__(self, quantile_range: Tuple[float, float] = (25.0, 75.0)) -> None:
        self.quantile_range = quantile_range

    def _stats(self, x2):
        q1, q3 = np.nanpercentile(x2, self.quantile_range, axis=0)
        return np.nanmedian(x2, 0), q3 - q1


@register("transform", "log", aliases=("log1p",))
class LogScaler(BaseTransform):
    """``log(x + offset)`` followed by standardisation.

    Useful for strictly positive, right-skewed data (prices, PV, wind).
    ``offset=None`` chooses a small offset so that ``min(x) + offset > 0``.
    """

    _state_attrs = ("offset_", "inner_")

    def __init__(self, offset: Optional[float] = None) -> None:
        self.offset = offset

    def fit(self, x, y=None):
        arr = np.asarray(x, dtype=float)
        if self.offset is None:
            mn = np.nanmin(arr)
            rng = np.nanmax(arr) - mn
            self.offset_ = float(max(0.0, -mn) + 1e-3 * (rng if rng > 0 else 1.0))
        else:
            self.offset_ = float(self.offset)
        self.inner_ = StandardScaler().fit(np.log(arr + self.offset_))
        return self

    def transform(self, x):
        arr = np.asarray(x, dtype=float)
        return self.inner_.transform(np.log(np.maximum(arr + self.offset_, 1e-12)))

    def inverse_transform(self, x):
        return np.exp(self.inner_.inverse_transform(x)) - self.offset_


@register("transform", "identity", aliases=("none",))
class IdentityScaler(BaseTransform):
    """No-op transform."""

    def __init__(self) -> None:
        pass

    def transform(self, x):
        return np.asarray(x, dtype=float).copy()


def make_scaler(name) -> BaseTransform:
    """Create a scaler from a name or return the given instance."""
    from ..registry import get

    if name is None or name is False:
        return IdentityScaler()
    if isinstance(name, BaseTransform):
        return name
    return get("transform", name)()

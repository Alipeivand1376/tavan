"""Input validation helpers (internal)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def to_numpy(x, name: str = "x", dtype=np.float64) -> np.ndarray:
    """Convert array-likes (lists, pandas objects, TimeSeries) to ndarray."""
    if x is None:
        raise ValueError(f"{name} must not be None")
    if hasattr(x, "values") and not isinstance(x, np.ndarray) and not isinstance(x, dict):
        vals = x.values
        x = vals() if callable(vals) else vals
    arr = np.asarray(x, dtype=dtype)
    return arr


def as_2d(x, name: str = "x") -> Tuple[np.ndarray, bool]:
    """Return ``(array_2d, was_1d)`` with shape ``(n_time, n_channels)``."""
    arr = to_numpy(x, name)
    if arr.ndim == 1:
        return arr[:, None], True
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {arr.shape}")
    return arr, False


def check_finite(x: np.ndarray, name: str = "x") -> np.ndarray:
    if not np.all(np.isfinite(x)):
        raise ValueError(
            f"{name} contains NaN or inf; impute missing values first "
            "(e.g. tavan.preprocess.Imputer or TimeSeries.impute())"
        )
    return x


def check_positive_int(v, name: str) -> int:
    iv = int(v)
    if iv <= 0 or iv != v:
        raise ValueError(f"{name} must be a positive integer, got {v!r}")
    return iv


def check_fraction(v: float, name: str, open_left: bool = True) -> float:
    fv = float(v)
    lo_ok = fv > 0 if open_left else fv >= 0
    if not (lo_ok and fv < 1):
        raise ValueError(f"{name} must lie in {'(0, 1)' if open_left else '[0, 1)'}, got {v!r}")
    return fv


def make_rng(seed: Optional[int | np.random.Generator]) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def check_is_fitted(obj, attr: str = "is_fitted_") -> None:
    if not getattr(obj, attr, False):
        raise RuntimeError(f"{type(obj).__name__} is not fitted yet; call fit() first")

"""Windowing utilities: supervised windows and periodic profiles."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .._validation import as_2d


def sliding_windows(x, lookback: int, horizon: int, stride: int = 1,
                    covariates: Optional[np.ndarray] = None):
    """Build supervised ``(input, target)`` windows.

    Parameters
    ----------
    x:
        Series of shape ``(T,)`` or ``(T, C)``.
    lookback, horizon:
        Input and output window lengths.
    stride:
        Step between consecutive windows.
    covariates:
        Optional ``(T, F)`` array; when given, the future covariates aligned
        with each target window are returned as a third element.

    Returns
    -------
    X : ndarray ``(n, lookback, C)``
    Y : ndarray ``(n, horizon, C)``
    F : ndarray ``(n, horizon, F)``, only if ``covariates`` is given.
    """
    arr, _ = as_2d(x)
    T = arr.shape[0]
    total = lookback + horizon
    if T < total:
        raise ValueError(f"series of length {T} is shorter than lookback+horizon={total}")
    win = np.lib.stride_tricks.sliding_window_view(arr, total, axis=0)[::stride]  # (n, C, total)
    win = np.ascontiguousarray(np.transpose(win, (0, 2, 1)))
    X, Y = win[:, :lookback], win[:, lookback:]
    if covariates is None:
        return X, Y
    cov, _ = as_2d(covariates)
    if cov.shape[0] != T:
        raise ValueError("covariates must have the same length as the series")
    cw = np.lib.stride_tricks.sliding_window_view(cov, total, axis=0)[::stride]
    cw = np.ascontiguousarray(np.transpose(cw, (0, 2, 1)))
    return X, Y, cw


def to_profiles(x, period: int, drop_incomplete: bool = True, offset: int = 0) -> np.ndarray:
    """Reshape a series into consecutive periods.

    ``(T, C) -> (T // period, period, C)``; e.g. hourly data with
    ``period=24`` gives one row per day.
    """
    arr, _ = as_2d(x)
    arr = arr[offset:]
    n = arr.shape[0] // period
    if n == 0:
        raise ValueError(f"series of length {arr.shape[0]} is shorter than one period ({period})")
    if not drop_incomplete and arr.shape[0] % period:
        pad = period - arr.shape[0] % period
        arr = np.concatenate([arr, np.full((pad, arr.shape[1]), np.nan)])
        n += 1
    return arr[: n * period].reshape(n, period, arr.shape[1])


def from_profiles(profiles: np.ndarray) -> np.ndarray:
    """Inverse of :func:`to_profiles` (concatenate periods)."""
    p = np.asarray(profiles)
    if p.ndim == 2:
        return p.reshape(-1)
    return p.reshape(-1, p.shape[-1])


def train_test_split(x, test: float | int = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """Chronological split (no shuffling)."""
    arr = np.asarray(x)
    n = arr.shape[0]
    k = int(round(n * test)) if isinstance(test, float) and test < 1 else int(test)
    if not 0 < k < n:
        raise ValueError("test size must leave both parts non-empty")
    return arr[: n - k], arr[n - k:]

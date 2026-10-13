"""Feature engineering: calendar encodings, Fourier terms and lags."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

from ..data.timeutils import day_of_week, day_of_year, hour_of_day, month_of_year, to_datetime64

CALENDAR = ("hour", "dow", "month", "doy", "weekend")


def calendar_features(timestamps, features: Sequence[str] = ("hour", "dow", "month"),
                      cyclical: bool = True) -> Tuple[np.ndarray, list]:
    """Encode timestamps as numeric features.

    Parameters
    ----------
    timestamps:
        Array of datetimes (any format accepted by ``numpy.datetime64``).
    features:
        Subset of ``("hour", "dow", "month", "doy", "weekend")``.
    cyclical:
        Encode periodic features as ``(sin, cos)`` pairs.

    Returns
    -------
    X : ndarray ``(T, F)``
    names : list of str
    """
    ts = to_datetime64(timestamps)
    cols, names = [], []
    spec = {
        "hour": (hour_of_day, 24.0),
        "dow": (day_of_week, 7.0),
        "month": (month_of_year, 12.0),
        "doy": (lambda t: day_of_year(t) - 1, 365.25),
    }
    for f in features:
        if f == "weekend":
            cols.append((day_of_week(ts) >= 5).astype(float))
            names.append("weekend")
            continue
        if f not in spec:
            raise ValueError(f"unknown calendar feature {f!r}; choose from {CALENDAR}")
        fn, period = spec[f]
        v = fn(ts).astype(float)
        if cyclical:
            ang = 2 * np.pi * v / period
            cols += [np.sin(ang), np.cos(ang)]
            names += [f"{f}_sin", f"{f}_cos"]
        else:
            cols.append(v)
            names.append(f)
    return np.column_stack(cols) if cols else np.zeros((len(ts), 0)), names


def fourier_terms(n: int, period: float, order: int = 3, start: int = 0) -> np.ndarray:
    """``2·order`` Fourier basis columns for a period (in steps)."""
    t = np.arange(start, start + n, dtype=float)
    cols = []
    for k in range(1, order + 1):
        ang = 2 * np.pi * k * t / period
        cols += [np.sin(ang), np.cos(ang)]
    return np.column_stack(cols)


def lag_matrix(x, lags: Iterable[int]) -> np.ndarray:
    """Stack lagged copies of a 1-D series (rows with missing lags are NaN)."""
    x = np.asarray(x, dtype=float).ravel()
    lags = list(lags)
    out = np.full((x.size, len(lags)), np.nan)
    for j, L in enumerate(lags):
        if L == 0:
            out[:, j] = x
        elif L > 0:
            out[L:, j] = x[:-L]
    return out


def day_type(timestamps) -> np.ndarray:
    """Integer day type per timestamp: 0 = weekday, 1 = Saturday, 2 = Sunday."""
    dow = day_of_week(to_datetime64(timestamps))
    return np.where(dow == 5, 1, np.where(dow == 6, 2, 0))


def season(timestamps, hemisphere: str = "north") -> np.ndarray:
    """Integer season: 0 winter, 1 spring, 2 summer, 3 autumn."""
    m = month_of_year(to_datetime64(timestamps))
    s = ((m + 1) % 12) // 3
    if hemisphere == "south":
        s = (s + 2) % 4
    return s


def one_hot(labels, n_classes: int | None = None) -> np.ndarray:
    labels = np.asarray(labels).astype(int).ravel()
    k = n_classes or int(labels.max()) + 1
    out = np.zeros((labels.size, k))
    out[np.arange(labels.size), labels] = 1.0
    return out

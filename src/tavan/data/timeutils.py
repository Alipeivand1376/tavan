"""Frequency strings and timestamp utilities based on ``numpy.datetime64``."""

from __future__ import annotations

import re
from typing import Optional, Union

import numpy as np

_UNITS = {
    "s": ("s", 1), "sec": ("s", 1), "second": ("s", 1),
    "min": ("m", 1), "t": ("m", 1), "m": ("m", 1), "minute": ("m", 1),
    "h": ("h", 1), "hour": ("h", 1), "hr": ("h", 1),
    "d": ("D", 1), "day": ("D", 1),
    "w": ("W", 1), "week": ("W", 1),
}

FreqLike = Union[str, np.timedelta64, None]


def parse_freq(freq: FreqLike) -> Optional[np.timedelta64]:
    """Convert ``"15min"``, ``"1h"``, ``"D"``... to ``numpy.timedelta64``.

    >>> parse_freq("15min")
    numpy.timedelta64(15,'m')
    """
    if freq is None:
        return None
    if isinstance(freq, np.timedelta64):
        return freq
    m = re.fullmatch(r"\s*(\d+)?\s*([A-Za-z]+)\s*", str(freq))
    if not m:
        raise ValueError(f"cannot parse frequency {freq!r}")
    n = int(m.group(1) or 1)
    unit = m.group(2).lower()
    if unit not in _UNITS and unit.endswith("s"):
        unit = unit[:-1]
    if unit not in _UNITS:
        raise ValueError(f"unknown frequency unit in {freq!r}; use s, min, h, D or W")
    code, mult = _UNITS[unit]
    return np.timedelta64(n * mult, code)


def freq_to_seconds(freq: FreqLike) -> Optional[float]:
    td = parse_freq(freq)
    return None if td is None else float(td / np.timedelta64(1, "s"))


def steps_per(freq: FreqLike, span: str = "1D") -> Optional[int]:
    """Number of ``freq`` steps in ``span`` (e.g. 24 hourly steps per day)."""
    s = freq_to_seconds(freq)
    if s is None:
        return None
    ratio = freq_to_seconds(span) / s
    if abs(ratio - round(ratio)) > 1e-9:
        return None
    return int(round(ratio))


def to_datetime64(values) -> np.ndarray:
    """Convert strings / datetimes / datetime64 to ``datetime64[s]``."""
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[s]")
    if hasattr(values, "to_numpy"):  # pandas
        return np.asarray(values.to_numpy(), dtype="datetime64[s]")
    out = np.array([np.datetime64(str(v).strip().replace(" ", "T"), "s") for v in arr.ravel()])
    return out.reshape(arr.shape)


def date_range(start, periods: int, freq: FreqLike) -> np.ndarray:
    """Regular timestamp grid."""
    td = parse_freq(freq)
    t0 = np.datetime64(str(start).replace(" ", "T"), "s")
    return t0 + np.arange(periods) * td.astype("timedelta64[s]")


def infer_freq(timestamps: np.ndarray) -> Optional[np.timedelta64]:
    """Most common spacing of a timestamp array."""
    ts = to_datetime64(timestamps)
    if ts.size < 2:
        return None
    d = np.diff(ts).astype("timedelta64[s]").astype(np.int64)
    vals, counts = np.unique(d, return_counts=True)
    step = int(vals[np.argmax(counts)])
    for unit, div in (("D", 86400), ("h", 3600), ("m", 60)):
        if step % div == 0:
            return np.timedelta64(step // div, unit)
    return np.timedelta64(step, "s")


def hour_of_day(ts: np.ndarray) -> np.ndarray:
    ts = ts.astype("datetime64[s]")
    return (ts - ts.astype("datetime64[D]")).astype("timedelta64[s]").astype(np.int64) / 3600.0


def day_of_week(ts: np.ndarray) -> np.ndarray:
    """Monday = 0 ... Sunday = 6."""
    days = ts.astype("datetime64[D]").astype(np.int64)
    return (days + 3) % 7


def month_of_year(ts: np.ndarray) -> np.ndarray:
    """January = 0 ... December = 11."""
    return ts.astype("datetime64[M]").astype(np.int64) % 12


def day_of_year(ts: np.ndarray) -> np.ndarray:
    """1-based day of year."""
    d = ts.astype("datetime64[D]")
    return (d - d.astype("datetime64[Y]")).astype(np.int64) + 1

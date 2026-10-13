"""Condition encoding for conditional scenario generation."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ..data.timeutils import day_of_week, day_of_year, month_of_year, to_datetime64

CALENDAR_CONDITIONS = ("dow", "weekend", "month", "season", "doy")


def encode_calendar(starts, names: Sequence[str], weekend: Sequence[int] = (5, 6)) -> Tuple[np.ndarray, list]:
    """Encode the start date of each profile as condition features.

    Parameters
    ----------
    starts:
        One timestamp per profile (e.g. the midnight of each day).
    names:
        Subset of ``dow`` (one-hot, 7), ``weekend`` (binary), ``month``
        (sin/cos), ``season`` (one-hot, 4), ``doy`` (sin/cos).
    weekend:
        Weekend days (Monday = 0).  Use ``(4,)`` for Friday weekends as in
        Iran, or ``(4, 5)`` for Friday–Saturday.
    """
    ts = to_datetime64(np.atleast_1d(starts))
    cols, labels = [], []
    for nme in names:
        if nme == "dow":
            d = day_of_week(ts)
            cols.append(np.eye(7)[d])
            labels += [f"dow_{i}" for i in range(7)]
        elif nme == "weekend":
            cols.append(np.isin(day_of_week(ts), list(weekend)).astype(float)[:, None])
            labels.append("weekend")
        elif nme == "month":
            a = 2 * np.pi * month_of_year(ts) / 12.0
            cols.append(np.column_stack([np.sin(a), np.cos(a)]))
            labels += ["month_sin", "month_cos"]
        elif nme == "season":
            s = ((month_of_year(ts) + 1) % 12) // 3
            cols.append(np.eye(4)[s])
            labels += [f"season_{i}" for i in range(4)]
        elif nme == "doy":
            a = 2 * np.pi * (day_of_year(ts) - 1) / 365.25
            cols.append(np.column_stack([np.sin(a), np.cos(a)]))
            labels += ["doy_sin", "doy_cos"]
        else:
            raise ValueError(f"unknown calendar condition {nme!r}; choose from {CALENDAR_CONDITIONS}")
    return (np.concatenate(cols, axis=1) if cols else np.zeros((len(ts), 0))), labels


def aggregate_to_profiles(cond, period: int, n_profiles: int, how: str = "mean") -> np.ndarray:
    """Aggregate a step-level condition ``(T, k)`` to one row per profile."""
    c = np.asarray(cond, dtype=float)
    if c.ndim == 1:
        c = c[:, None]
    if c.shape[0] == n_profiles:
        return c
    blocks = c[: n_profiles * period].reshape(n_profiles, period, c.shape[1])
    if how == "mean":
        return blocks.mean(1)
    if how == "max":
        return blocks.max(1)
    if how == "min":
        return blocks.min(1)
    if how == "flatten":
        return blocks.reshape(n_profiles, -1)
    raise ValueError("how must be 'mean', 'max', 'min' or 'flatten'")


def parse_conditions(conditions) -> Tuple[Sequence[str], bool]:
    """Return ``(calendar_names, use_custom)`` from the user specification."""
    if conditions is None or conditions is False:
        return (), False
    if isinstance(conditions, str):
        conditions = [conditions]
    cal, custom = [], False
    for c in conditions:
        if c == "calendar":
            cal += ["weekend", "doy"]
        elif c == "custom":
            custom = True
        elif c in CALENDAR_CONDITIONS:
            cal.append(c)
        else:
            raise ValueError(f"unknown condition {c!r}; use {CALENDAR_CONDITIONS}, 'calendar' or 'custom'")
    seen, ordered = set(), []
    for c in cal:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return tuple(ordered), custom


def date_steps(date, period: int, freq: np.timedelta64) -> np.ndarray:
    """Timestamps of one profile starting at ``date``."""
    t0 = to_datetime64(np.atleast_1d(date))[0]
    return t0 + np.arange(period) * freq.astype("timedelta64[s]")


def _maybe(x: Optional[np.ndarray]) -> Optional[np.ndarray]:
    return None if x is None else np.asarray(x, dtype=float)

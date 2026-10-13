"""Physics-aware constraints for generated scenarios.

Generic generative models know nothing about physics: a GAN can emit
negative demand, PV output at midnight or impossible ramps.  Constraints
are *projections* applied to scenarios in physical units, and every
constraint can also *measure* how often raw samples violate it, which is a
useful quality metric in its own right.

All constraints act on arrays of shape ``(n_scenarios, horizon, channels)``
and accept an optional ``context`` dict (e.g. per-step ``timestamps`` of
shape ``(n, horizon)``).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np

from ..base import BaseEstimator
from ..registry import register

Channel = Union[int, str, None]


class Constraint(BaseEstimator):
    """Base class.  Sub-classes implement :meth:`project` and :meth:`violation`."""

    channel: Channel = None

    def _cols(self, X: np.ndarray, names: Optional[Sequence[str]]):
        if self.channel is None:
            return slice(None)
        if isinstance(self.channel, str):
            if names is None or self.channel not in names:
                raise KeyError(f"unknown channel {self.channel!r}")
            return [list(names).index(self.channel)]
        return [int(self.channel)]

    def project(self, X: np.ndarray, context: Optional[Dict] = None, names=None) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def violation(self, X: np.ndarray, context: Optional[Dict] = None, names=None, tol: float = 1e-9) -> np.ndarray:
        """Boolean ``(n,)``: does each scenario violate the constraint?"""
        P = self.project(X.copy(), context, names)
        return np.any(np.abs(P - X) > tol * (1 + np.abs(X)), axis=(1, 2))

    def __call__(self, X, context=None, names=None):
        return self.project(X, context, names)


@register("constraint", "bounds")
class Bounds(Constraint):
    """Clip values to ``[lower, upper]`` (scalars or per-step arrays)."""

    def __init__(self, lower=None, upper=None, channel: Channel = None) -> None:
        self.lower = lower
        self.upper = upper
        self.channel = channel

    def project(self, X, context=None, names=None):
        c = self._cols(X, names)
        lo = None if self.lower is None else np.asarray(self.lower, float).reshape(1, -1, 1) if np.ndim(self.lower) else self.lower
        hi = None if self.upper is None else np.asarray(self.upper, float).reshape(1, -1, 1) if np.ndim(self.upper) else self.upper
        X[:, :, c] = np.clip(X[:, :, c], lo, hi)
        return X


@register("constraint", "nonnegative", aliases=("non_negative", "positive"))
class NonNegative(Bounds):
    """Values must be ≥ 0 (demand, generation, prices in most markets)."""

    def __init__(self, channel: Channel = None) -> None:
        super().__init__(lower=0.0, upper=None, channel=channel)

    def get_params(self):
        return {"channel": self.channel}


@register("constraint", "ramp", aliases=("ramp_limit",))
class RampLimit(Constraint):
    """Limit step-to-step changes: ``-max_down ≤ x[t] - x[t-1] ≤ max_up``.

    The projection is applied forward in time, so the first value of each
    scenario is kept.
    """

    def __init__(self, max_up: float, max_down: Optional[float] = None, channel: Channel = None) -> None:
        self.max_up = max_up
        self.max_down = max_down
        self.channel = channel

    def project(self, X, context=None, names=None):
        c = self._cols(X, names)
        down = self.max_up if self.max_down is None else self.max_down
        sub = X[:, :, c]
        for t in range(1, sub.shape[1]):
            sub[:, t] = np.clip(sub[:, t], sub[:, t - 1] - down, sub[:, t - 1] + self.max_up)
        X[:, :, c] = sub
        return X


@register("constraint", "envelope", aliases=("cap",))
class EnvelopeCap(Constraint):
    """Per-step upper (and optional lower) envelope of shape ``(horizon,)``."""

    def __init__(self, upper, lower=None, channel: Channel = None) -> None:
        self.upper = upper
        self.lower = lower
        self.channel = channel

    def project(self, X, context=None, names=None):
        c = self._cols(X, names)
        hi = np.asarray(self.upper, float).reshape(1, -1, 1)
        lo = None if self.lower is None else np.asarray(self.lower, float).reshape(1, -1, 1)
        X[:, :, c] = np.clip(X[:, :, c], lo, hi)
        return X


@register("constraint", "clear_sky", aliases=("clearsky", "pv_envelope"))
class ClearSkyCap(Constraint):
    """PV output bounded by the clear-sky envelope and zero at night.

    Requires per-step timestamps in ``context["timestamps"]`` — provided
    automatically by :class:`~tavan.generate.ScenarioGenerator` when sampling
    with a ``date``.

    Parameters
    ----------
    latitude, longitude, tz_offset:
        Site location and UTC offset of the timestamps (hours).
    capacity:
        Installed capacity in the scenario's units.
    margin:
        Multiplicative tolerance on the envelope (cloud-edge enhancement can
        briefly exceed clear-sky irradiance).
    """

    def __init__(self, latitude: float, longitude: float, capacity: float = 1.0, tz_offset: float = 0.0,
                 margin: float = 1.05, derate: float = 0.85, freq_minutes: Optional[float] = None,
                 channel: Channel = None) -> None:
        self.latitude = latitude
        self.longitude = longitude
        self.capacity = capacity
        self.tz_offset = tz_offset
        self.margin = margin
        self.derate = derate
        self.freq_minutes = freq_minutes
        self.channel = channel

    def envelope(self, timestamps) -> np.ndarray:
        from ..power.solar import clear_sky_pv

        ts = np.asarray(timestamps)
        env = clear_sky_pv(ts.ravel(), self.latitude, self.longitude, self.capacity, self.tz_offset,
                           self.derate, self.freq_minutes)
        return np.minimum(env.reshape(ts.shape) * self.margin, self.capacity)

    def project(self, X, context=None, names=None):
        if not context or context.get("timestamps") is None:
            raise ValueError("ClearSkyCap needs timestamps: sample with `date=` or pass context={'timestamps': ...}")
        ts = np.asarray(context["timestamps"])
        if ts.ndim == 1:
            ts = np.broadcast_to(ts, X.shape[:2])
        env = self.envelope(ts)[:, :, None]
        c = self._cols(X, names)
        X[:, :, c] = np.clip(X[:, :, c], 0.0, env)
        return X


@register("constraint", "energy", aliases=("energy_budget",))
class EnergyBudget(Constraint):
    """Rescale each scenario so that its total lies in ``[min_total, max_total]``."""

    def __init__(self, min_total: Optional[float] = None, max_total: Optional[float] = None,
                 channel: Channel = None) -> None:
        self.min_total = min_total
        self.max_total = max_total
        self.channel = channel

    def project(self, X, context=None, names=None):
        c = self._cols(X, names)
        sub = X[:, :, c]
        tot = sub.sum(axis=1, keepdims=True)
        target = np.clip(tot, self.min_total, self.max_total)
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(np.abs(tot) > 1e-12, target / tot, 1.0)
        X[:, :, c] = sub * factor
        return X


class Custom(Constraint):
    """Wrap a function ``fn(X, context) -> X`` as a constraint (not saveable)."""

    def __init__(self, fn: Callable[[np.ndarray, Optional[Dict]], np.ndarray], name: str = "custom") -> None:
        self.fn = fn
        self.name = name

    def project(self, X, context=None, names=None):
        return np.asarray(self.fn(X, context), dtype=float)


def apply_constraints(X: np.ndarray, constraints: Sequence[Constraint], context: Optional[Dict] = None,
                      names=None, n_sweeps: int = 1) -> np.ndarray:
    """Apply constraints in order (optionally several sweeps for interacting ones)."""
    X = np.array(X, dtype=float, copy=True)
    squeeze = X.ndim == 2
    if squeeze:
        X = X[:, :, None]
    for _ in range(max(1, n_sweeps)):
        for c in constraints:
            X = c.project(X, context, names)
    return X[:, :, 0] if squeeze else X


def violation_rates(X: np.ndarray, constraints: Sequence[Constraint], context: Optional[Dict] = None,
                    names=None) -> Dict[str, float]:
    """Fraction of scenarios violating each constraint."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[:, :, None]
    out = {}
    for i, c in enumerate(constraints):
        key = f"{type(c).__name__}" + (f"[{c.channel}]" if getattr(c, "channel", None) is not None else "")
        if key in out:
            key = f"{key}#{i}"
        out[key] = float(np.mean(c.violation(X, context, names)))
    return out

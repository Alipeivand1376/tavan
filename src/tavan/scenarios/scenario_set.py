"""The :class:`ScenarioSet` container."""

from __future__ import annotations

import csv
import json
from typing import Dict, Optional, Sequence, Union

import numpy as np

from .._stats import weighted_quantile
from .reduction import kantorovich_distance, reduce_scenarios
from .risk import LossSpec, conditional_value_at_risk, risk_summary, scenario_loss, value_at_risk


class ScenarioSet:
    """A weighted set of scenarios in physical units.

    Parameters
    ----------
    values:
        ``(n_scenarios, horizon)`` or ``(n_scenarios, horizon, channels)``.
    probabilities:
        Scenario probabilities (uniform if omitted).
    timestamps:
        Optional ``(horizon,)`` timestamps shared by all scenarios.
    names, unit:
        Channel names and physical unit.

    Examples
    --------
    >>> import numpy as np
    >>> from tavan.scenarios import ScenarioSet
    >>> s = ScenarioSet(np.random.default_rng(0).normal(size=(500, 24)))
    >>> small = s.reduce(10, method="forward")
    >>> len(small), round(small.probabilities.sum(), 6)
    (10, 1.0)
    """

    def __init__(self, values, probabilities=None, timestamps=None, names: Optional[Sequence[str]] = None,
                 unit: str = "", meta: Optional[Dict] = None) -> None:
        v = np.asarray(values, dtype=float)
        if v.ndim == 2:
            v = v[:, :, None]
        if v.ndim != 3:
            raise ValueError("values must have shape (n, horizon[, channels])")
        self.values = v
        n = v.shape[0]
        if probabilities is None:
            p = np.full(n, 1.0 / n)
        else:
            p = np.asarray(probabilities, dtype=float)
            if p.shape != (n,) or np.any(p < 0) or p.sum() <= 0:
                raise ValueError("probabilities must be non-negative, one per scenario")
            p = p / p.sum()
        self.probabilities = p
        self.timestamps = None if timestamps is None else np.asarray(timestamps)
        self.names = list(names) if names is not None else [f"x{i}" for i in range(v.shape[2])]
        self.unit = unit
        self.meta = dict(meta or {})

    # ----------------------------------------------------------- basics
    @property
    def n_scenarios(self) -> int:
        return self.values.shape[0]

    @property
    def horizon(self) -> int:
        return self.values.shape[1]

    @property
    def n_channels(self) -> int:
        return self.values.shape[2]

    def __len__(self) -> int:
        return self.n_scenarios

    def __repr__(self) -> str:
        return (f"ScenarioSet(n_scenarios={self.n_scenarios}, horizon={self.horizon}, "
                f"channels={self.names}, unit={self.unit!r})")

    def __getitem__(self, idx) -> "ScenarioSet":
        if isinstance(idx, (int, np.integer)):
            idx = [int(idx)]
        return self._subset(np.arange(self.n_scenarios)[idx])

    def _subset(self, idx, probs=None) -> "ScenarioSet":
        idx = np.asarray(idx, dtype=int)
        p = self.probabilities[idx] if probs is None else probs
        return ScenarioSet(self.values[idx], p, self.timestamps, self.names, self.unit, self.meta)

    def to_numpy(self, squeeze: bool = True) -> np.ndarray:
        """``(n, H)`` for single-channel sets, otherwise ``(n, H, C)``."""
        return self.values[:, :, 0].copy() if squeeze and self.n_channels == 1 else self.values.copy()

    def channel(self, name: Union[str, int]) -> "ScenarioSet":
        c = self.names.index(name) if isinstance(name, str) else int(name)
        return ScenarioSet(self.values[:, :, [c]], self.probabilities, self.timestamps, [self.names[c]],
                           self.unit, self.meta)

    # ------------------------------------------------------- statistics
    def mean(self) -> np.ndarray:
        """Probability-weighted mean trajectory ``(H, C)``."""
        return np.tensordot(self.probabilities, self.values, axes=1)

    def std(self) -> np.ndarray:
        mu = self.mean()
        return np.sqrt(np.tensordot(self.probabilities, (self.values - mu) ** 2, axes=1))

    def quantiles(self, q=(0.05, 0.25, 0.5, 0.75, 0.95)) -> np.ndarray:
        """Weighted quantiles, shape ``(len(q), H, C)``."""
        return np.asarray(weighted_quantile(self.values, list(np.atleast_1d(q)), self.probabilities, axis=0))

    def losses(self, loss: LossSpec = "sum", channel: Optional[Union[int, str]] = None) -> np.ndarray:
        """Per-scenario scalar loss (see :func:`tavan.scenarios.scenario_loss`)."""
        c = self.names.index(channel) if isinstance(channel, str) else channel
        return scenario_loss(self.values, loss, c)

    def var(self, alpha: float = 0.95, loss: LossSpec = "sum", channel=None) -> float:
        """Value-at-Risk of the scenario loss."""
        return value_at_risk(self.losses(loss, channel), alpha, self.probabilities)

    def cvar(self, alpha: float = 0.95, loss: LossSpec = "sum", channel=None) -> float:
        """Conditional Value-at-Risk (expected shortfall) of the scenario loss."""
        return conditional_value_at_risk(self.losses(loss, channel), alpha, self.probabilities)

    def risk_summary(self, loss: LossSpec = "sum", channel=None, alphas=(0.9, 0.95, 0.99)) -> dict:
        return risk_summary(self.losses(loss, channel), self.probabilities, alphas)

    def tail(self, alpha: float = 0.95, loss: LossSpec = "sum", channel=None) -> "ScenarioSet":
        """Scenarios with loss ≥ VaR (renormalised probabilities)."""
        L = self.losses(loss, channel)
        v = value_at_risk(L, alpha, self.probabilities)
        idx = np.where(L >= v)[0]
        return self._subset(idx)

    # -------------------------------------------------------- reduction
    def reduce(self, k: int, method: str = "kmedoids", alpha: float = 0.95, loss: LossSpec = "sum",
               channel=None, seed: Optional[int] = 0, **kwargs) -> "ScenarioSet":
        """Reduce to ``k`` representative scenarios.

        ``method``: ``"kmedoids"``, ``"forward"``, ``"cvar"`` (tail-preserving)
        or ``"random"``.  The Kantorovich distance to the original set is
        stored in ``meta["reduction_distance"]``.
        """
        losses = self.losses(loss, channel) if method in ("cvar", "risk", "tail") else None
        idx, p = reduce_scenarios(self.values, k, method, self.probabilities, losses, alpha, seed, **kwargs)
        out = self._subset(idx, p)
        out.meta = dict(self.meta, reduced_from=self.n_scenarios, reduction_method=method,
                        reduction_distance=kantorovich_distance(self.values, self.probabilities, idx))
        return out

    # ------------------------------------------------------- evaluation
    def evaluate(self, reference, metrics=None) -> Dict[str, float]:
        """Compare with real samples ``(m, H[, C])`` (see :func:`evaluate_scenarios`)."""
        from .metrics import DEFAULT_METRICS, evaluate_scenarios

        ref = reference.values if isinstance(reference, ScenarioSet) else np.asarray(reference, dtype=float)
        if self.n_scenarios > 0 and not np.allclose(self.probabilities, self.probabilities[0]):
            rng = np.random.default_rng(0)
            gen = self.values[rng.choice(self.n_scenarios, size=max(500, self.n_scenarios), p=self.probabilities)]
        else:
            gen = self.values
        return evaluate_scenarios(ref, gen, metrics or DEFAULT_METRICS)

    # --------------------------------------------------------------- I/O
    def to_dict(self) -> dict:
        return {
            "values": self.values.tolist(),
            "probabilities": self.probabilities.tolist(),
            "timestamps": None if self.timestamps is None else [str(t) for t in self.timestamps],
            "names": self.names,
            "unit": self.unit,
        }

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)

    @classmethod
    def from_json(cls, path: str) -> "ScenarioSet":
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        ts = None if d["timestamps"] is None else np.array(d["timestamps"], dtype="datetime64[s]")
        return cls(d["values"], d["probabilities"], ts, d["names"], d["unit"])

    def to_csv(self, path: str) -> None:
        """Long format: scenario, probability, step, timestamp, channels..."""
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["scenario", "probability", "step", "timestamp", *self.names])
            for s in range(self.n_scenarios):
                for t in range(self.horizon):
                    ts = str(self.timestamps[t]) if self.timestamps is not None else ""
                    w.writerow([s, f"{self.probabilities[s]:.10g}", t, ts, *[f"{v:.10g}" for v in self.values[s, t]]])

    def to_optimization(self) -> dict:
        """Plain dict for stochastic-programming models (Pyomo, CVXPY, JuMP...).

        Keys: ``scenarios`` (list of ids), ``probability`` {s: p},
        ``value`` {(s, t, channel): v}.
        """
        ids = list(range(self.n_scenarios))
        vals = {(s, t, self.names[c]): float(self.values[s, t, c])
                for s in ids for t in range(self.horizon) for c in range(self.n_channels)}
        return {"scenarios": ids, "periods": list(range(self.horizon)), "channels": self.names,
                "probability": {s: float(self.probabilities[s]) for s in ids}, "value": vals}

    # ----------------------------------------------------------- plots
    def plot(self, reference=None, ax=None, **kwargs):
        """Fan chart of quantile bands with sample paths (requires matplotlib)."""
        from ..viz import plot_scenarios

        return plot_scenarios(self, reference=reference, ax=ax, **kwargs)

    def report(self, path: str = "tavan_report.html", reference=None, title: str = "Scenario report",
               loss: LossSpec = "sum") -> str:
        """Write a self-contained HTML report (no plotting libraries needed)."""
        from ..report import scenario_report

        return scenario_report(self, path=path, reference=reference, title=title, loss=loss)

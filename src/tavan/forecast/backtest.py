"""Walk-forward backtesting and simple hyper-parameter search.

Honest time-series evaluation never shuffles: the model is repeatedly
trained on the past and tested on the next block (rolling or expanding
origin).  :func:`backtest` returns per-fold and aggregate metrics and always
reports seasonal-naive skill so you can see whether a model earns its
complexity.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .._validation import as_2d
from .baselines import SeasonalNaive
from .metrics import evaluate_point


@dataclass
class BacktestResult:
    """Aggregated walk-forward results."""

    metrics: Dict[str, float]
    fold_metrics: List[Dict[str, float]]
    predictions: List[np.ndarray]
    actuals: List[np.ndarray]
    origins: List[int]
    skill: Dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0

    def __repr__(self) -> str:
        m = ", ".join(f"{k}={v:.4g}" for k, v in self.metrics.items())
        return f"BacktestResult(folds={len(self.fold_metrics)}, {m})"

    def summary(self) -> Dict[str, Any]:
        return {"metrics": self.metrics, "skill_vs_snaive": self.skill, "n_folds": len(self.fold_metrics),
                "seconds": self.seconds}


def backtest(model, y, horizon: int, n_folds: int = 5, step: Optional[int] = None,
             expanding: bool = True, initial: Optional[int] = None, X_exog=None,
             metrics: Sequence[str] = ("mae", "rmse", "smape"), season_length: int = 1,
             refit: bool = True, verbose: bool = False) -> BacktestResult:
    """Walk-forward backtest of a forecaster.

    Parameters
    ----------
    model:
        A :class:`BaseForecaster` (or anything with ``clone``/``fit``/``predict``).
    y:
        Series ``(T,)`` or ``(T, C)``.
    horizon:
        Forecast length evaluated at each origin.
    n_folds:
        Number of test origins.
    step:
        Spacing between origins (default: ``horizon``).
    expanding:
        Expanding window (use all history) vs rolling window of fixed size.
    initial:
        Size of the first training window (default: half the series).
    refit:
        Refit the model at each origin (``False`` fits once and only updates
        the context — much faster for expensive models).
    """
    arr, _ = as_2d(y, "y")
    T = len(arr)
    step = step or horizon
    initial = initial or max(model.__dict__.get("lookback", 0) + horizon, T // 2)
    origins = list(range(initial, T - horizon + 1, step))
    if len(origins) > n_folds:
        origins = origins[-n_folds:]
    if not origins:
        raise ValueError("series too short for the requested backtest configuration")

    preds, acts, folds = [], [], []
    fitted = None
    t0 = time.perf_counter()
    for i, o in enumerate(origins):
        train_start = 0 if expanding else max(0, o - initial)
        ytr = arr[train_start:o]
        xtr = None if X_exog is None else X_exog[train_start:o]
        if refit or fitted is None:
            fitted = model.clone() if hasattr(model, "clone") else model
            fitted.fit(ytr, xtr)
        xf = None if X_exog is None else X_exog[o:o + horizon]
        fc = fitted.predict(ytr, xf, horizon=horizon).mean.reshape(horizon, -1)
        actual = arr[o:o + horizon]
        preds.append(fc)
        acts.append(actual)
        fm = evaluate_point(actual, fc, metrics, y_train=ytr, season_length=season_length)
        folds.append(fm)
        if verbose:
            print(f"fold {i + 1}/{len(origins)} origin={o} " + " ".join(f"{k}={v:.3g}" for k, v in fm.items()))

    agg = {k: float(np.mean([f[k] for f in folds])) for k in folds[0]}
    # seasonal-naive skill
    snaive = SeasonalNaive(season_length=max(season_length, 1), horizon=horizon)
    sk_preds = []
    for o in origins:
        snaive.fit(arr[:o])
        sk_preds.append(snaive.predict(arr[:o], horizon=horizon).mean.reshape(horizon, -1))
    skill = {}
    for k in agg:
        base = np.mean([list(evaluate_point(a, p, [k]).values())[0] for a, p in zip(acts, sk_preds)])
        skill[k] = float(1 - agg[k] / base) if base > 0 else float("nan")
    return BacktestResult(agg, folds, preds, acts, origins, skill, time.perf_counter() - t0)


def grid_search(model_factory: Callable[..., Any], param_grid: Dict[str, Sequence], y, horizon: int,
                n_folds: int = 3, metric: str = "rmse", X_exog=None, season_length: int = 1,
                verbose: bool = False) -> Tuple[Dict[str, Any], "BacktestResult", List[Tuple[Dict, float]]]:
    """Exhaustive walk-forward hyper-parameter search.

    ``model_factory(**params)`` builds a forecaster.  Returns the best
    parameters, their backtest result and the full ranked list.
    """
    keys = list(param_grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*(param_grid[k] for k in keys))]
    results = []
    best = None
    for params in combos:
        model = model_factory(**params)
        res = backtest(model, y, horizon, n_folds=n_folds, X_exog=X_exog, metrics=(metric,),
                       season_length=season_length)
        score = res.metrics[metric]
        results.append((params, score))
        if verbose:
            print(f"{params} -> {metric}={score:.4g}")
        if best is None or score < best[1]:
            best = (params, score, res)
    results.sort(key=lambda t: t[1])
    return best[0], best[2], results

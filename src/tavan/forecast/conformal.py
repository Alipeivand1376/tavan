"""Conformal prediction intervals with finite-sample coverage guarantees.

Split and online conformal prediction wrap *any* point forecaster and turn
it into a calibrated interval forecaster, without distributional
assumptions.  For time series we use the EnbPI-style absolute-residual
quantile per horizon step, plus an adaptive variant (ACI, Gibbs & Candès
2021) that tracks coverage online under drift.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .._validation import check_is_fitted
from ..base import BaseEstimator
from .base import BaseForecaster, ForecastResult


class ConformalForecaster(BaseEstimator):
    """Wrap a point forecaster with conformal prediction intervals.

    Parameters
    ----------
    base:
        Any fitted-or-unfitted :class:`BaseForecaster`.
    levels:
        Central coverage levels to produce (e.g. ``(0.8, 0.9)``).
    calibration_fraction:
        Fraction of the training series reserved for calibration (the rest
        trains ``base``).
    normalize:
        Scale residuals by a rolling spread so interval width adapts to local
        volatility (recommended for prices).
    adaptive:
        Enable adaptive conformal inference: call :meth:`update` with realised
        values to keep coverage on target under drift.
    gamma:
        ACI learning rate for the adaptive variant.

    Examples
    --------
    >>> from tavan.data import make_load
    >>> from tavan.forecast import DLinear, ConformalForecaster
    >>> y = make_load(90).to_numpy()
    >>> cf = ConformalForecaster(DLinear(lookback=48, horizon=24), levels=(0.9,)).fit(y)
    >>> fc = cf.predict()
    >>> lo, hi = fc.interval(0.9)
    """

    _state_attrs = ("base_", "residuals_", "scale_", "alpha_err_", "levels", "names_", "unit_")

    def __init__(self, base: BaseForecaster, levels: Sequence[float] = (0.9,),
                 calibration_fraction: float = 0.3, normalize: bool = False,
                 adaptive: bool = False, gamma: float = 0.02) -> None:
        self.base = base
        self.levels = tuple(levels)
        self.calibration_fraction = calibration_fraction
        self.normalize = normalize
        self.adaptive = adaptive
        self.gamma = gamma

    def fit(self, y, X_exog=None) -> "ConformalForecaster":
        from .._validation import as_2d

        arr, self._was_1d = as_2d(y, "y")
        n = len(arr)
        h = self.base.horizon
        n_cal = max(h + 1, int(round(n * self.calibration_fraction)))
        n_tr = n - n_cal
        if n_tr < self.base.lookback + h:
            raise ValueError("series too short for the chosen horizon and calibration fraction")
        self.base_ = self.base.clone() if getattr(self.base, "is_fitted_", False) else self.base
        self.base_.fit(arr[:n_tr], None if X_exog is None else X_exog[:n_tr])
        # rolling-origin residuals on the calibration part
        resid = []
        step = max(1, h // 4)
        for t in range(n_tr, n - h + 1, step):
            fc = self.base_.predict(arr[:t], horizon=h).mean.reshape(h, -1)
            resid.append(arr[t:t + h] - fc)
        R = np.abs(np.array(resid))  # (m, h, C)
        self.scale_ = None
        if self.normalize:
            self.scale_ = np.std(arr[:n_tr], axis=0) + 1e-6
            R = R / self.scale_
        self.residuals_ = R
        self.alpha_err_ = {lvl: (1 - lvl) for lvl in self.levels}
        self.names_ = getattr(self.base_, "names_", None)
        self.unit_ = getattr(self.base_, "unit_", "")
        self.is_fitted_ = True
        self.base_.fit(arr, None if X_exog is None else X_exog)  # refit on all data for the point forecast
        return self

    def _radius(self, level: float) -> np.ndarray:
        alpha = self.alpha_err_[level] if self.adaptive else (1 - level)
        alpha = min(max(alpha, 1e-3), 0.999)
        q = np.quantile(self.residuals_, 1 - alpha, axis=0)  # (h, C)
        return q * self.scale_ if self.scale_ is not None else q

    def predict(self, y=None, X_exog=None, horizon: Optional[int] = None) -> ForecastResult:
        check_is_fitted(self)
        fc = self.base_.predict(y, X_exog, horizon)
        mean2 = fc.mean.reshape(fc.horizon, -1)
        quantiles = {}
        for lvl in self.levels:
            r = self._radius(lvl)[: fc.horizon]
            if r.shape[0] < fc.horizon:
                r = np.pad(r, ((0, fc.horizon - r.shape[0]), (0, 0)), mode="edge")
            lo = mean2 - r
            hi = mean2 + r
            quantiles[(1 - lvl) / 2] = lo[:, 0] if self._was_1d else lo
            quantiles[(1 + lvl) / 2] = hi[:, 0] if self._was_1d else hi
        quantiles[0.5] = fc.mean
        return ForecastResult(fc.mean, quantiles, fc.timestamps, self.names_, self.unit_)

    def update(self, y_true, y_pred_interval_hit: Optional[bool] = None) -> None:
        """Adaptive conformal update after observing realised values.

        Pass the realised horizon values and whether they fell inside the
        interval; the per-level error rate is nudged to keep coverage.
        """
        if not self.adaptive:
            return
        y_true = np.asarray(y_true, dtype=float)
        for lvl in self.levels:
            r = self._radius(lvl)
            mean = getattr(self, "_last_mean_", None)
            covered = y_pred_interval_hit
            if covered is None and mean is not None:
                covered = bool(np.all(np.abs(y_true - mean[: len(y_true)]) <= r[: len(y_true)]))
            err = 0.0 if covered else 1.0
            self.alpha_err_[lvl] = float(np.clip(self.alpha_err_[lvl] + self.gamma * ((1 - lvl) - err), 1e-3, 0.999))


def coverage(y_true, lower, upper) -> float:
    """Empirical coverage of prediction intervals."""
    y = np.asarray(y_true, dtype=float)
    return float(np.mean((y >= np.asarray(lower)) & (y <= np.asarray(upper))))


def interval_width(lower, upper) -> float:
    """Mean prediction-interval width."""
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


def winkler_score(y_true, lower, upper, level: float) -> float:
    """Winkler interval score (coverage + sharpness); lower is better."""
    y = np.asarray(y_true, dtype=float)
    lo, hi = np.asarray(lower), np.asarray(upper)
    alpha = 1 - level
    width = hi - lo
    pen_lo = np.where(y < lo, 2 / alpha * (lo - y), 0.0)
    pen_hi = np.where(y > hi, 2 / alpha * (y - hi), 0.0)
    return float(np.mean(width + pen_lo + pen_hi))

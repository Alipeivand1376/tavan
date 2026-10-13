"""Base classes for point and probabilistic forecasters.

Forecasters operate on a univariate or multivariate series and predict the
next ``horizon`` steps from the previous ``lookback`` steps.  Probabilistic
forecasters additionally return quantiles.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .._validation import as_2d, check_finite, check_is_fitted, make_rng
from ..base import BaseEstimator


class ForecastResult:
    """Container for a forecast.

    Attributes
    ----------
    mean : ndarray ``(horizon,)`` or ``(horizon, C)``
        Point forecast.
    quantiles : dict {level: ndarray} or None
        Predictive quantiles keyed by probability level.
    timestamps : ndarray or None
        Timestamps of the forecast horizon.
    """

    def __init__(self, mean, quantiles=None, timestamps=None, names=None, unit: str = ""):
        self.mean = np.asarray(mean, dtype=float)
        self.quantiles = quantiles
        self.timestamps = timestamps
        self.names = names
        self.unit = unit

    @property
    def horizon(self) -> int:
        return self.mean.shape[0]

    def interval(self, level: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
        """Return the central ``level`` prediction interval ``(lower, upper)``."""
        if not self.quantiles:
            raise ValueError("no quantiles available; use a probabilistic forecaster or conformal intervals")
        lo_lvl, hi_lvl = (1 - level) / 2, (1 + level) / 2
        return self._q(lo_lvl), self._q(hi_lvl)

    def _q(self, level: float) -> np.ndarray:
        levels = np.array(sorted(self.quantiles))
        if np.any(np.isclose(levels, level)):
            return self.quantiles[levels[np.argmin(np.abs(levels - level))]]
        qs = np.stack([self.quantiles[l] for l in levels])
        return np.array([np.interp(level, levels, qs[:, i]) for i in range(qs.shape[1])]) if qs.ndim == 2 else \
            np.stack([[np.interp(level, levels, qs[:, t, c]) for c in range(qs.shape[2])] for t in range(qs.shape[1])])

    def to_scenarioset(self, n: int = 200, seed: int = 0):
        """Sample scenarios consistent with the predictive quantiles."""
        from ..scenarios import ScenarioSet

        rng = make_rng(seed)
        u = rng.random((n, self.horizon))
        if self.quantiles:
            levels = np.array(sorted(self.quantiles))
            qs = np.stack([np.atleast_2d(self.quantiles[l].reshape(self.horizon, -1)) for l in levels])
            out = np.empty((n, self.horizon, qs.shape[2]))
            for t in range(self.horizon):
                for c in range(qs.shape[2]):
                    out[:, t, c] = np.interp(u[:, t], levels, qs[:, t, c])
        else:
            out = np.repeat(self.mean.reshape(1, self.horizon, -1), n, 0)
        return ScenarioSet(out, timestamps=self.timestamps, names=self.names, unit=self.unit)

    def plot(self, history=None, ax=None, **kwargs):
        from ..viz import plot_forecast

        return plot_forecast(self, history=history, ax=ax, **kwargs)

    def __repr__(self) -> str:
        q = f", quantiles={sorted(self.quantiles)}" if self.quantiles else ""
        return f"ForecastResult(horizon={self.horizon}, channels={self.mean.shape[1] if self.mean.ndim > 1 else 1}{q})"


class BaseForecaster(BaseEstimator):
    """Interface of all forecasters.

    Sub-classes set ``requires_windows`` and implement either the
    window-based ``_fit``/``_predict_batch`` (when ``requires_windows``) or
    the series-based ``_fit_series``/``_forecast``.
    """

    requires_windows: bool = True
    supports_quantiles: bool = False
    supports_exog: bool = False

    def __init__(self, lookback: int = 0, horizon: int = 1) -> None:
        self.lookback = lookback
        self.horizon = horizon

    # ------------------------------------------------------------- fit
    def fit(self, y, X_exog=None) -> "BaseForecaster":
        """Fit on a 1-D/2-D series ``y`` with optional exogenous inputs ``X_exog``."""
        arr, self._was_1d = as_2d(y, "y")
        check_finite(arr, "y")
        self.n_channels_ = arr.shape[1]
        self.y_train_ = arr
        if X_exog is not None:
            if not self.supports_exog:
                raise ValueError(f"{type(self).__name__} does not support exogenous inputs")
            X_exog = check_finite(np.asarray(X_exog, dtype=float), "X_exog")
            self.n_exog_ = X_exog.shape[1] if X_exog.ndim > 1 else 1
        else:
            self.n_exog_ = 0
        self._fit_impl(arr, X_exog)
        self.is_fitted_ = True
        return self

    def predict(self, y=None, X_exog=None, horizon: Optional[int] = None) -> ForecastResult:
        """Forecast the next ``horizon`` steps following ``y`` (defaults to training tail)."""
        check_is_fitted(self)
        h = horizon or self.horizon
        context = self.y_train_ if y is None else as_2d(y, "y")[0]
        res = self._forecast_impl(context, X_exog, h)
        if self._was_1d and res.mean.ndim == 2 and res.mean.shape[1] == 1:
            res.mean = res.mean[:, 0]
            if res.quantiles:
                res.quantiles = {k: v[:, 0] if v.ndim == 2 else v for k, v in res.quantiles.items()}
        return res

    # --------------------------------------------- windowing utilities
    def _make_windows(self, y: np.ndarray, X_exog=None):
        from ..data.windows import sliding_windows

        if X_exog is not None:
            Xw, Yw, Fw = sliding_windows(y, self.lookback, self.horizon, covariates=X_exog)
            return Xw, Yw, Fw
        Xw, Yw = sliding_windows(y, self.lookback, self.horizon)
        return Xw, Yw, None

    # ------------------------------------------------------- overrides
    def _fit_impl(self, y: np.ndarray, X_exog):  # pragma: no cover
        raise NotImplementedError

    def _forecast_impl(self, context: np.ndarray, X_exog, horizon: int) -> ForecastResult:  # pragma: no cover
        raise NotImplementedError

    def _wrap(self, mean, quantiles=None) -> ForecastResult:
        return ForecastResult(mean, quantiles, names=getattr(self, "names_", None), unit=getattr(self, "unit_", ""))

    # --------------------------------------------------- persistence
    _COMMON_STATE = ("n_channels_", "n_exog_", "_was_1d", "is_fitted_", "y_train_")

    def _get_state(self):
        st = {a: getattr(self, a) for a in self._COMMON_STATE if hasattr(self, a)}
        for a in self._state_attrs:
            if hasattr(self, a):
                st[a] = getattr(self, a)
        return st

    def _set_state(self, state):
        for k, v in state.items():
            setattr(self, k, v)

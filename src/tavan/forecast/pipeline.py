"""High-level :class:`Forecaster` tying preprocessing to any base model."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from .._validation import check_is_fitted
from ..base import BaseEstimator
from ..data.timeseries import TimeSeries, as_timeseries
from ..registry import get
from .backtest import backtest
from .base import BaseForecaster, ForecastResult
from .conformal import ConformalForecaster


class Forecaster(BaseEstimator):
    """User-facing forecasting pipeline.

    Wraps imputation, optional calendar covariates, a base model and
    optional conformal intervals, and returns forecasts in physical units
    with timestamps.

    Parameters
    ----------
    model:
        Name (``"dlinear"``, ``"nlinear"``, ``"ngrc"``, ``"nhits"``,
        ``"tide"``, ``"tsmixer"``, ``"patchtst"``, ``"seasonal_naive"`` …) or
        a :class:`BaseForecaster` instance.
    lookback, horizon:
        Window lengths (ignored if a model instance is given).
    impute:
        Imputation method for missing inputs, or ``None``.
    calendar:
        Calendar covariates to add as exogenous inputs (requires timestamps
        and a model with ``supports_exog``), e.g. ``("hour", "dow")``.
    intervals:
        ``None`` for point forecasts, or coverage levels (e.g. ``(0.9,)``)
        to wrap the model in conformal prediction.
    model_params:
        Extra keyword arguments for the base model.

    Examples
    --------
    >>> from tavan.data import make_load
    >>> from tavan.forecast import Forecaster
    >>> f = Forecaster("dlinear", lookback=72, horizon=24, intervals=(0.9,)).fit(make_load(120))
    >>> fc = f.predict()
    >>> lo, hi = fc.interval(0.9)
    """

    _state_attrs = ("model_", "imputer_", "period_", "names_", "unit_", "freq_", "last_ts_",
                    "calendar_", "is_fitted_")

    def __init__(self, model: Union[str, BaseForecaster] = "dlinear", lookback: int = 48, horizon: int = 24,
                 impute: Optional[str] = "auto", calendar: Optional[Sequence[str]] = None,
                 intervals: Optional[Sequence[float]] = None, calibration_fraction: float = 0.3,
                 model_params: Optional[Dict[str, Any]] = None, seed: int = 0, verbose: int = 0) -> None:
        self.model = model
        self.lookback = lookback
        self.horizon = horizon
        self.impute = impute
        self.calendar = tuple(calendar) if calendar else None
        self.intervals = tuple(intervals) if intervals else None
        self.calibration_fraction = calibration_fraction
        self.model_params = model_params
        self.seed = seed
        self.verbose = verbose

    def _make_model(self) -> BaseForecaster:
        if isinstance(self.model, BaseForecaster):
            return self.model
        cls = get("forecaster", self.model)
        params = dict(self.model_params or {})
        names = cls._param_names()
        for k in ("lookback", "horizon"):
            if k in names and k not in params:
                params[k] = getattr(self, k)
        for k in ("seed", "verbose"):
            if k in names and k not in params:
                params[k] = getattr(self, k)
        return cls(**params)

    def _calendar_exog(self, ts: TimeSeries, n: int, future_start=None, horizon=0):
        if not self.calendar:
            return None, None
        from ..preprocess.features import calendar_features

        if ts.timestamps is None:
            raise ValueError("calendar covariates require timestamps")
        hist, _ = calendar_features(ts.timestamps, self.calendar)
        fut = None
        if horizon and ts.freq is not None:
            start = ts.timestamps[-1] + ts.freq.astype("timedelta64[s]")
            future = start + np.arange(horizon) * ts.freq.astype("timedelta64[s]")
            fut, _ = calendar_features(future, self.calendar)
        return hist, fut

    def fit(self, data, timestamps=None, freq=None) -> "Forecaster":
        ts = as_timeseries(data, freq=freq, timestamps=timestamps)
        self.period_ = ts.steps_per_day
        self.names_ = list(ts.names)
        self.unit_ = ts.unit
        self.freq_ = ts.freq
        self.last_ts_ = None if ts.timestamps is None else ts.timestamps[-1]
        values = ts.values
        self.imputer_ = None
        if self.impute and not np.all(np.isfinite(values)):
            from ..preprocess.impute import Imputer

            self.imputer_ = Imputer(self.impute, period=self.period_)
            values = self.imputer_.fit_transform(values)
        exog, _ = self._calendar_exog(ts, len(values))
        self.calendar_ = exog is not None
        base = self._make_model()
        if self.intervals:
            self.model_ = ConformalForecaster(base, levels=self.intervals,
                                              calibration_fraction=self.calibration_fraction)
            self.model_.fit(values if values.shape[1] > 1 else values[:, 0], exog)
        else:
            self.model_ = base
            self.model_.fit(values if values.shape[1] > 1 else values[:, 0], exog)
        self._train_ts_ = ts
        self.is_fitted_ = True
        return self

    def predict(self, horizon: Optional[int] = None, future_exog=None) -> ForecastResult:
        """Forecast ahead; attaches horizon timestamps when the freq is known."""
        check_is_fitted(self)
        h = horizon or self.horizon
        exog = future_exog
        if self.calendar_ and exog is None:
            _, exog = self._calendar_exog(self._train_ts_, 0, horizon=h)
        fc = self.model_.predict(X_exog=exog, horizon=h)
        if self.last_ts_ is not None and self.freq_ is not None:
            step = self.freq_.astype("timedelta64[s]")
            fc.timestamps = self.last_ts_ + np.arange(1, h + 1) * step
        fc.names, fc.unit = self.names_, self.unit_
        return fc

    def backtest(self, data=None, horizon: Optional[int] = None, n_folds: int = 5, **kwargs):
        """Walk-forward backtest of the configured model (point forecasts)."""
        ts = self._train_ts_ if data is None else as_timeseries(data)
        base = self._make_model()
        season = self.period_ or 1
        y = ts.values if ts.n_channels > 1 else ts.values[:, 0]
        return backtest(base, y, horizon or self.horizon, n_folds=n_folds, season_length=season, **kwargs)

    @property
    def history(self):
        base = getattr(self.model_, "base_", self.model_)
        return getattr(base, "history_", None)

"""Baseline forecasters — always compare new models against these."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..registry import register
from .base import BaseForecaster


@register("forecaster", "naive", aliases=("last",))
class Naive(BaseForecaster):
    """Repeat the last observed value across the horizon."""

    requires_windows = False

    def __init__(self, horizon: int = 1) -> None:
        super().__init__(lookback=1, horizon=horizon)

    def _fit_impl(self, y, X_exog):
        pass

    def _forecast_impl(self, context, X_exog, horizon):
        return self._wrap(np.repeat(context[-1:], horizon, axis=0))


@register("forecaster", "seasonal_naive", aliases=("snaive",))
class SeasonalNaive(BaseForecaster):
    """Repeat the value from one season ago (the standard seasonal baseline)."""

    requires_windows = False

    def __init__(self, season_length: int = 24, horizon: int = 1) -> None:
        super().__init__(lookback=season_length, horizon=horizon)
        self.season_length = season_length

    def _fit_impl(self, y, X_exog):
        pass

    def _forecast_impl(self, context, X_exog, horizon):
        m = self.season_length
        if len(context) < m:
            return self._wrap(np.repeat(context[-1:], horizon, axis=0))
        season = context[-m:]
        idx = np.arange(horizon) % m
        return self._wrap(season[idx])


@register("forecaster", "drift")
class Drift(BaseForecaster):
    """Linear extrapolation of the average trend (Hyndman's drift method)."""

    requires_windows = False

    def __init__(self, horizon: int = 1) -> None:
        super().__init__(lookback=2, horizon=horizon)

    def _fit_impl(self, y, X_exog):
        pass

    def _forecast_impl(self, context, X_exog, horizon):
        slope = (context[-1] - context[0]) / max(len(context) - 1, 1)
        steps = np.arange(1, horizon + 1)[:, None]
        return self._wrap(context[-1] + steps * slope)


@register("forecaster", "mean")
class MeanForecast(BaseForecaster):
    """Predict the historical mean (optionally over a trailing window)."""

    requires_windows = False

    def __init__(self, window: Optional[int] = None, horizon: int = 1) -> None:
        super().__init__(lookback=1, horizon=horizon)
        self.window = window

    def _fit_impl(self, y, X_exog):
        pass

    def _forecast_impl(self, context, X_exog, horizon):
        ctx = context if self.window is None else context[-self.window:]
        return self._wrap(np.repeat(ctx.mean(0, keepdims=True), horizon, axis=0))

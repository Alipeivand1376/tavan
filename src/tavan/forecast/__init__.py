"""Time-series forecasting with modern, non-recurrent models.

Baselines
    ``naive``, ``seasonal_naive``, ``drift``, ``mean``
Linear (closed-form ridge, millisecond training)
    ``dlinear``, ``nlinear``
Reservoir computing
    ``ngrc`` — next-generation reservoir / nonlinear vector autoregression
Neural (shared NumPy autograd, CPU, no RNNs)
    ``nhits``, ``tide``, ``tsmixer``, ``patchtst``

All neural and linear models support multivariate series, direct multi-step
forecasting, exogenous/calendar covariates and quantile (pinball) outputs.
:class:`ConformalForecaster` adds distribution-free prediction intervals to
any point model; :class:`Forecaster` is the high-level pipeline.
"""

from .backtest import BacktestResult, backtest, grid_search
from .base import BaseForecaster, ForecastResult
from .baselines import Drift, MeanForecast, Naive, SeasonalNaive
from .conformal import ConformalForecaster, coverage, interval_width, winkler_score
from .linear import DLinear, NLinear
from .metrics import (
    evaluate_point,
    mae,
    mape,
    mase,
    mse,
    pinball_loss,
    quantile_coverage,
    r2,
    rmse,
    smape,
)
from .neural import NHiTS, PatchTST, TiDE, TSMixer
from .pipeline import Forecaster
from .reservoir import NGRC

__all__ = [
    "Forecaster", "BaseForecaster", "ForecastResult", "Naive", "SeasonalNaive", "Drift", "MeanForecast",
    "DLinear", "NLinear", "NGRC", "NHiTS", "TiDE", "TSMixer", "PatchTST", "ConformalForecaster",
    "coverage", "interval_width", "winkler_score", "backtest", "grid_search", "BacktestResult",
    "evaluate_point", "mae", "mse", "rmse", "mape", "smape", "mase", "r2", "pinball_loss",
    "quantile_coverage",
]

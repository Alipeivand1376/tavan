"""tavan — lightweight time-series scenario generation and forecasting.

``tavan`` (Persian: *power/energy*) is a dependency-light Python library for
probabilistic time-series work in power systems, energy and finance.  Its
only hard dependency is NumPy: the deep-learning models run on a small
built-in automatic-differentiation engine, on CPU.

Two main capabilities
---------------------
**Scenario generation** — fit a generative model (diffusion, VAE, normalising
flow, GAN, Gaussian mixture, copula or bootstrap) to historical profiles and
draw conditional, physically-constrained scenarios, with CVaR-aware sampling,
scenario reduction and quality metrics.

**Forecasting** — modern, non-recurrent point and probabilistic forecasters
(DLinear, NLinear, NG-RC, N-HiTS, TiDE, TSMixer, PatchTST) with conformal
prediction intervals and honest walk-forward backtesting.

Plus a small power-systems toolkit (DC power flow, state estimation,
false-data-injection studies, moving-target defense) and offline demo data.

Quick start
-----------
>>> import tavan
>>> ts = tavan.datasets.make_load(120)                 # demo demand series
>>> gen = tavan.ScenarioGenerator("gmm", conditions=["weekend"]).fit(ts)
>>> scen = gen.sample(200, date="2024-06-01")
>>> scen.cvar(0.95, loss="max")                        # peak-demand risk
>>> fc = tavan.Forecaster("dlinear", intervals=(0.9,)).fit(ts).predict()
"""

from . import data as datasets  # noqa: F401
from . import forecast, generate, nn, power, preprocess, recipes, scenarios  # noqa: F401
from ._version import __version__
from .base import load
from .data import (
    TimeSeries,
    make_load,
    make_price,
    make_pv,
    make_wind,
    sliding_windows,
    to_profiles,
)
from .forecast import (
    NGRC,
    ConformalForecaster,
    DLinear,
    Forecaster,
    NHiTS,
    NLinear,
    PatchTST,
    TiDE,
    TSMixer,
    backtest,
)
from .generate import (
    CGAN,
    VAE,
    Bootstrap,
    Diffusion,
    GaussianCopula,
    GaussianMixture,
    RealNVP,
    ScenarioGenerator,
)
from .registry import available, register
from .scenarios import ScenarioSet, conditional_value_at_risk, cvar, value_at_risk

__all__ = [
    "__version__", "load", "available", "register", "datasets",
    "TimeSeries", "make_load", "make_pv", "make_wind", "make_price", "sliding_windows", "to_profiles",
    "ScenarioGenerator", "ScenarioSet", "Bootstrap", "GaussianCopula", "GaussianMixture", "VAE",
    "Diffusion", "RealNVP", "CGAN", "cvar", "value_at_risk", "conditional_value_at_risk",
    "Forecaster", "DLinear", "NLinear", "NGRC", "NHiTS", "TiDE", "TSMixer", "PatchTST",
    "ConformalForecaster", "backtest",
    "nn", "data", "preprocess", "generate", "scenarios", "forecast", "power", "recipes",
]

from . import data  # noqa: E402,F401

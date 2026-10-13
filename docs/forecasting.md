# Forecasting

Modern, non-recurrent point and probabilistic forecasters.

```python
from tavan.forecast import Forecaster, backtest, DLinear

# high-level pipeline with calibrated intervals
f = Forecaster("nhits", lookback=96, horizon=24, intervals=(0.9,)).fit(ts)
fc = f.predict()
lo, hi = fc.interval(0.9)

# honest walk-forward backtest with seasonal-naive skill
r = backtest(DLinear(96, 24), ts.to_numpy(), horizon=24, n_folds=5, season_length=24)
print(r.metrics, r.skill)
```

## Models

| family | names |
|--------|-------|
| baselines | `naive`, `seasonal_naive`, `drift`, `mean` |
| linear (ridge, instant) | `dlinear`, `nlinear` |
| reservoir | `ngrc` |
| neural | `nhits`, `tide`, `tsmixer`, `patchtst` |

All support multivariate series, exogenous/calendar covariates and quantile
(pinball) outputs:

```python
m = NHiTS(96, 24, quantiles=(0.1, 0.5, 0.9)).fit(ts.values)
```

## Conformal intervals

`ConformalForecaster` wraps any point model with distribution-free prediction
intervals, including an adaptive variant that tracks coverage under drift.

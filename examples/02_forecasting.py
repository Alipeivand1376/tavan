"""Example 2 — Probabilistic forecasting with honest backtesting.

Compare a few forecasters against the seasonal-naive baseline with
walk-forward backtesting, then produce a calibrated 90% interval forecast
with conformal prediction.

Run:  python examples/02_forecasting.py
"""

import tavan
from tavan.forecast import ConformalForecaster, DLinear, NGRC, NHiTS, SeasonalNaive, backtest, coverage

y = tavan.make_load(n_days=150, seed=0).to_numpy()
H, L = 24, 96

# Fast models are backtested over 5 folds; the neural model refits each fold,
# so we give it fewer epochs here to keep the example quick.
print("Walk-forward backtest (skill vs seasonal-naive):")
for name, model, folds in [
    ("seasonal-naive", SeasonalNaive(24, H), 5),
    ("DLinear", DLinear(L, H), 5),
    ("NG-RC", NGRC(L, H, delays=6), 5),
    ("N-HiTS", NHiTS(L, H, epochs=60), 3),
]:
    r = backtest(model, y, horizon=H, n_folds=folds, season_length=24)
    print(f"  {name:15s} MAE={r.metrics['mae']:6.3f}  skill={r.skill['mae']:+.1%}")

# Calibrated prediction intervals with conformal prediction (any point model)
cf = ConformalForecaster(DLinear(L, H), levels=(0.9,), calibration_fraction=0.3).fit(y[: -24 * 15])
hits = []
for o in range(len(y) - 24 * 15, len(y) - H, H):      # roll over the last ~2 weeks
    lo, hi = cf.predict(y[:o], horizon=H).interval(0.9)
    hits.append(coverage(y[o:o + H], lo, hi))
print(f"\n90% interval empirical coverage (rolling over {len(hits)} days):",
      round(float(sum(hits) / len(hits)), 3))

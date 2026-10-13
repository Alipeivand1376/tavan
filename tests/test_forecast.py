"""Tests for forecasters, conformal intervals and backtesting."""

import numpy as np
import pytest

import tavan
from tavan.forecast import (
    NGRC,
    ConformalForecaster,
    DLinear,
    Forecaster,
    NHiTS,
    NLinear,
    SeasonalNaive,
    backtest,
    coverage,
)
from tavan.forecast.metrics import evaluate_point

rng = np.random.default_rng(0)


def _signal(n=24 * 100):
    t = np.arange(n)
    amp = 1 + 0.4 * np.sin(2 * np.pi * t / (24 * 30))
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.6 * e[i - 1] + rng.normal(0, 0.12)
    return 10 + amp * np.sin(2 * np.pi * t / 24) + 0.3 * np.sin(2 * np.pi * t / 12) + e


@pytest.mark.parametrize("cls,kw", [
    (SeasonalNaive, {"season_length": 24, "horizon": 24}),
    (DLinear, {"lookback": 96, "horizon": 24}),
    (NLinear, {"lookback": 96, "horizon": 24}),
    (NGRC, {"lookback": 96, "horizon": 24, "delays": 6}),
])
def test_forecaster_shapes(cls, kw):
    y = _signal()
    m = cls(**kw).fit(y[:-24])
    fc = m.predict(horizon=24)
    assert fc.mean.shape == (24,)
    assert np.isfinite(fc.mean).all()


def test_linear_beats_seasonal_naive_on_learnable_signal():
    y = _signal()
    r_dl = backtest(DLinear(96, 24), y, 24, n_folds=4, season_length=24)
    assert r_dl.skill["mae"] > 0.05  # >5% better than seasonal naive


def test_neural_forecaster_learns():
    # amplitude drifts over time, so a global model should clearly beat naive
    y = _signal(24 * 90)
    m = NHiTS(96, 24, epochs=150).fit(y[:-24])
    fc = m.predict(horizon=24)
    # the learned forecast is far better than persistence and close to the signal floor
    naive = np.repeat(y[-25], 24)
    assert evaluate_point(y[-24:], fc.mean)["mae"] < 0.6 * evaluate_point(y[-24:], naive)["mae"]
    assert evaluate_point(y[-24:], fc.mean)["mae"] < 0.3  # good absolute accuracy


def test_conformal_coverage():
    y = _signal()
    cf = ConformalForecaster(DLinear(96, 24), levels=(0.9,), calibration_fraction=0.3).fit(y[:-24])
    # evaluate coverage over several origins
    hits = []
    for o in range(len(y) - 24 * 20, len(y) - 24, 24):
        fc = cf.predict(y[:o], horizon=24)
        lo, hi = fc.interval(0.9)
        hits.append(coverage(y[o:o + 24], lo, hi))
    assert 0.80 <= np.mean(hits) <= 1.0


def test_quantile_forecaster():
    y = _signal(24 * 60)
    m = NHiTS(96, 24, quantiles=(0.1, 0.5, 0.9), epochs=80).fit(y)
    fc = m.predict(horizon=24)
    assert set(fc.quantiles) >= {0.1, 0.5, 0.9}
    # quantiles should be ordered on average
    assert np.mean(fc.quantiles[0.1]) <= np.mean(fc.quantiles[0.9])


def test_backtest_skill_structure():
    y = _signal()
    r = backtest(DLinear(96, 24), y, 24, n_folds=3, season_length=24)
    assert "mae" in r.metrics and "mae" in r.skill
    assert len(r.fold_metrics) == 3


def test_multivariate_forecast():
    ts = tavan.make_load(60, with_temperature=True, seed=2)
    m = DLinear(48, 12).fit(ts.values)
    fc = m.predict(horizon=12)
    assert fc.mean.shape == (12, 2)


def test_pipeline_attaches_timestamps():
    ts = tavan.make_load(60, seed=1)
    f = Forecaster("dlinear", lookback=72, horizon=24).fit(ts)
    fc = f.predict()
    assert fc.timestamps is not None and len(fc.timestamps) == 24


def test_forecaster_save_load(tmp_path):
    y = _signal(24 * 50)
    m = DLinear(96, 24).fit(y)
    a = m.predict(horizon=24).mean
    p = m.save(str(tmp_path / "f.npz"))
    b = tavan.load(p).predict(y, horizon=24).mean
    assert np.allclose(a, b)

"""Tests for data container and preprocessing."""

import numpy as np
import pytest

from tavan.data import TimeSeries, make_load, make_pv, parse_freq, sliding_windows
from tavan.preprocess import (
    Imputer,
    OutlierCleaner,
    Pipeline,
    RobustScaler,
    StandardScaler,
    calendar_features,
    impute,
)

rng = np.random.default_rng(0)


@pytest.mark.parametrize("s,expected", [("15min", (15, "m")), ("1h", (1, "h")), ("H", (1, "h")),
                                        ("2hours", (2, "h")), ("D", (1, "D"))])
def test_parse_freq(s, expected):
    td = parse_freq(s)
    assert td == np.timedelta64(*expected)


def test_timeseries_basics():
    ts = TimeSeries(rng.random(48), start="2025-01-01", freq="1h")
    assert ts.n_steps == 48 and ts.steps_per_day == 24
    assert ts.to_profiles(24).shape == (2, 24, 1)
    assert ts.profile_timestamps(24).shape == (2,)


def test_windows_shapes():
    X, Y = sliding_windows(np.arange(100.0), 10, 5)
    assert X.shape == (86, 10, 1) and Y.shape == (86, 5, 1)
    assert np.allclose(X[0, :, 0], np.arange(10))
    assert np.allclose(Y[0, :, 0], np.arange(10, 15))


def test_linear_imputation_exact():
    x = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    assert np.allclose(impute(x, "linear"), [1, 2, 3, 4, 5])


@pytest.mark.parametrize("method", ["linear", "ffill", "bfill", "mean", "seasonal", "knn", "em"])
def test_imputers_fill_everything(method):
    clean = make_load(60, seed=1).values
    corrupt = clean.copy()
    miss = rng.random(clean.shape) < 0.05
    corrupt[miss] = np.nan
    out = Imputer(method, period=24).fit_transform(corrupt)
    assert np.isfinite(out).all()
    assert np.allclose(out[~miss], clean[~miss])


def test_seasonal_beats_mean_on_gaps():
    clean = make_load(90, seed=2).values
    corrupt = clean.copy()
    miss = rng.random(clean.shape) < 0.08
    corrupt[miss] = np.nan
    err_seasonal = np.sqrt(np.mean((Imputer("seasonal", period=24).fit_transform(corrupt)[miss] - clean[miss]) ** 2))
    err_mean = np.sqrt(np.mean((Imputer("mean").fit_transform(corrupt)[miss] - clean[miss]) ** 2))
    assert err_seasonal < err_mean


def test_scalers_roundtrip():
    x = rng.normal(5, 3, size=(100, 2))
    for sc in (StandardScaler(), RobustScaler()):
        z = sc.fit_transform(x)
        assert np.allclose(sc.inverse_transform(z), x, atol=1e-9)


def test_standard_scaler_normalizes():
    z = StandardScaler().fit_transform(rng.normal(5, 3, size=(500, 2)))
    assert np.allclose(z.mean(0), 0, atol=1e-9) and np.allclose(z.std(0), 1, atol=1e-6)


def test_outlier_detection():
    x = rng.normal(0, 1, 200)
    x[50] = 20.0
    oc = OutlierCleaner("hampel", action="nan")
    oc.transform(x)
    assert oc.mask_[50]


def test_pipeline_roundtrip():
    x = make_load(30, seed=3, missing_rate=0.05).values
    pipe = Pipeline([OutlierCleaner(), Imputer("seasonal", period=24), StandardScaler()])
    z = pipe.fit_transform(x)
    assert np.isfinite(z).all()


def test_calendar_features():
    ts = make_load(10)
    feats, names = calendar_features(ts.timestamps, ("hour", "dow"))
    assert feats.shape[0] == ts.n_steps and feats.shape[1] == len(names)
    assert np.all(np.abs(feats) <= 1.0 + 1e-9)


def test_pv_is_zero_at_night():
    pv = make_pv(10, latitude=35.7, longitude=51.4, tz_offset=3.5, seed=0)
    hours = (pv.timestamps.astype("datetime64[h]").astype(int) % 24)
    assert np.all(pv.values[(hours < 4) | (hours > 21), 0] < 1e-6)

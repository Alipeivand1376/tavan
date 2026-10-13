"""Tests for generative models and the scenario pipeline."""

import numpy as np
import pytest

import tavan
from tavan.data import make_load
from tavan.generate import (
    Bootstrap,
    ClearSkyCap,
    GaussianCopula,
    GaussianMixture,
    NonNegative,
    RampLimit,
    RealNVP,
    ScenarioGenerator,
    apply_constraints,
    violation_rates,
)
from tavan.scenarios import evaluate_scenarios

rng = np.random.default_rng(0)


def _profiles(seed=0):
    return make_load(120, seed=seed).to_profiles()[:, :, 0]


@pytest.mark.parametrize("cls,kw", [
    (Bootstrap, {}),
    (GaussianCopula, {}),
    (GaussianMixture, {"n_components": 3}),
])
def test_statistical_generators_shapes(cls, kw):
    X = _profiles()
    g = cls(**kw).fit(X)
    S = g.sample(50)
    assert S.shape == (50, X.shape[1])
    assert np.isfinite(S).all()


def test_gmm_likelihood_and_cluster():
    X = _profiles()
    g = GaussianMixture(n_components=4).fit(X)
    ll = g.score_samples(X)
    assert ll.shape == (len(X),) and np.isfinite(ll).all()
    labels = g.predict_component(X)
    assert set(np.unique(labels)).issubset(set(range(g.n_components_)))


def test_flow_exact_likelihood():
    X = _profiles()
    g = RealNVP(n_layers=4, epochs=80).fit(X)
    ll = g.score_samples(X[:10])
    assert ll.shape == (10,) and np.isfinite(ll).all()


@pytest.mark.parametrize("model,kw", [
    ("gmm", {}), ("copula", {}), ("bootstrap", {}),
    ("vae", {"epochs": 150}), ("flow", {"epochs": 60}), ("diffusion", {"epochs": 80}),
])
def test_generators_match_distribution(model, kw):
    X = _profiles()
    g = tavan.generate.ScenarioGenerator(model, scale="standard", model_params=kw).fit(X[:, :, None])
    S = g.sample(300).to_numpy()
    m = evaluate_scenarios(X, S, ["mmd", "correlation_error"])
    # VAE is a smoother density model; the others match more tightly
    assert m["mmd"] < (0.25 if model == "vae" else 0.2)


def test_conditional_generation_respects_condition():
    ts = make_load(200, seed=0)
    g = ScenarioGenerator("gmm", conditions=["weekend"]).fit(ts)
    weekday = g.sample(300, date="2024-06-03").to_numpy()   # Monday
    weekend = g.sample(300, date="2024-06-01").to_numpy()   # Saturday
    # weekends have lower demand in the generator
    assert weekend.mean() < weekday.mean()


def test_constraints_enforced():
    X = rng.normal(size=(100, 24, 1)) * 10
    out = apply_constraints(X, [NonNegative()])
    assert (out >= 0).all()
    rl = apply_constraints(X, [RampLimit(1.0)])
    assert np.abs(np.diff(rl, axis=1)).max() <= 1.0 + 1e-9


def test_clear_sky_cap_zero_at_night():
    from tavan.data.timeutils import date_range
    from tavan.power.solar import clear_sky_pv
    ts = date_range("2024-06-01", 24, "1h")
    cap = ClearSkyCap(35.7, 51.4, capacity=1.0, tz_offset=3.5)
    X = np.full((5, 24, 1), 2.0)
    out = cap.project(X.copy(), {"timestamps": np.broadcast_to(ts, (5, 24))})
    env = clear_sky_pv(ts, 35.7, 51.4, 1.0, 3.5)
    assert np.all(out[:, env < 1e-6, 0] < 1e-6)


def test_violation_rates_detect_raw_violations():
    X = rng.normal(size=(200, 24, 1))  # has negatives
    vr = violation_rates(X, [NonNegative()])
    assert list(vr.values())[0] > 0.5


def test_save_load_roundtrip(tmp_path):
    ts = make_load(90, seed=0)
    g = ScenarioGenerator("gmm", conditions=["weekend"], constraints=[NonNegative()]).fit(ts)
    s1 = g.sample(100, date="2024-06-01", seed=5).values
    p = g.save(str(tmp_path / "gen.npz"))
    g2 = tavan.load(p)
    s2 = g2.sample(100, date="2024-06-01", seed=5).values
    assert np.allclose(s1, s2)


def _stationary_profiles(n, seed, shift=0.0):
    r = np.random.default_rng(seed)
    t = np.arange(24)
    base = np.sin(2 * np.pi * t / 24) + 0.3 * np.sin(2 * np.pi * t / 12) + shift
    return (base[None, :] + r.normal(0, 0.1, size=(n, 24)))[:, :, None]


def test_drift_monitor():
    # stationary process: a fresh sample is NOT drift; a shifted one IS
    g = ScenarioGenerator("gmm", scale="standard").fit(_stationary_profiles(300, 0))
    same = g.monitor(_stationary_profiles(60, 1), refit=False)
    drifted = g.monitor(_stationary_profiles(60, 2, shift=1.5), refit=False)
    assert drifted.drift and not same.drift

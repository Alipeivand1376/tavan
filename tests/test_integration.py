"""End-to-end and cross-cutting tests."""

import numpy as np
import pytest

import tavan
from tavan.registry import available


def test_public_api_imports():
    for name in ["TimeSeries", "ScenarioGenerator", "Forecaster", "make_load", "cvar", "load"]:
        assert hasattr(tavan, name)


def test_registry_populated():
    reg = available()
    assert "diffusion" in reg["generator"]
    assert "dlinear" in reg["forecaster"]
    assert "nhits" in reg["forecaster"]
    assert "imputer" in reg["transform"]


def test_registry_create_by_name():
    from tavan.registry import create
    g = create("generator", "gmm", n_components=2)
    assert g.__class__.__name__ == "GaussianMixture"


def test_end_to_end_load_scenarios():
    from tavan.recipes import load_scenarios
    ts = tavan.make_load(120, seed=0)
    scen = load_scenarios(ts, n=100, model="gmm", date="2024-07-15", ramp_limit=40)
    assert scen.values.shape == (100, 24, 1)
    assert (scen.values >= 0).all()
    assert scen.cvar(0.95, loss="max") >= scen.var(0.95, loss="max") - 1e-9


def test_end_to_end_report(tmp_path):
    from tavan.recipes import load_scenarios
    scen = load_scenarios(tavan.make_load(90, seed=1), n=120, model="gmm", date="2024-05-01")
    p = scen.report(str(tmp_path / "r.html"))
    content = open(p, encoding="utf-8").read()
    assert "<svg" in content and "CVaR" in content and len(content) > 5000


def test_forecast_to_scenarios_bridge():
    y = tavan.make_load(80, seed=2).to_numpy()
    m = tavan.forecast.NHiTS(96, 24, quantiles=(0.1, 0.5, 0.9), epochs=60).fit(y)
    fc = m.predict(horizon=24)
    ss = fc.to_scenarioset(n=100)
    assert ss.values.shape[0] == 100 and ss.horizon == 24


def test_plugin_registration():
    from tavan.generate import BaseGenerator
    from tavan.registry import get, register

    @register("generator", "_test_plugin")
    class _Dummy(BaseGenerator):
        def __init__(self, seed=0):
            self.seed = seed

        def _fit(self, X, cond):
            self.mu_ = X.mean(0)

        def _sample(self, n, cond, rng):
            return np.repeat(self.mu_[None], n, 0)

    _Dummy._state_attrs = ("mu_", "n_features_", "n_cond_", "is_fitted_")
    g = get("generator", "_test_plugin")(seed=1).fit(np.random.rand(10, 4))
    assert g.sample(3).shape == (3, 4)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reproducible_sampling(seed):
    g = tavan.ScenarioGenerator("gmm").fit(tavan.make_load(60, seed=0))
    a = g.sample(50, seed=seed).values
    b = g.sample(50, seed=seed).values
    assert np.allclose(a, b)

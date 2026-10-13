"""Tests for scenario sets, risk measures, reduction and metrics."""

import numpy as np
import pytest

from tavan.scenarios import (
    ScenarioSet,
    conditional_value_at_risk,
    energy_score,
    fast_forward_selection,
    reduce_scenarios,
    value_at_risk,
)

rng = np.random.default_rng(0)


def test_var_cvar_known_distribution():
    # losses uniform on [0,1]: VaR_0.9 ~ 0.9, CVaR_0.9 ~ 0.95
    losses = np.linspace(0, 1, 10001)
    assert abs(value_at_risk(losses, 0.9) - 0.9) < 0.01
    assert abs(conditional_value_at_risk(losses, 0.9) - 0.95) < 0.01


def test_cvar_ge_var():
    losses = rng.normal(size=5000)
    for a in (0.8, 0.9, 0.95, 0.99):
        assert conditional_value_at_risk(losses, a) >= value_at_risk(losses, a) - 1e-9


def test_scenarioset_stats():
    V = rng.normal(size=(500, 24))
    s = ScenarioSet(V)
    assert s.mean().shape == (24, 1)
    q = s.quantiles([0.1, 0.5, 0.9])
    assert q.shape == (3, 24, 1)
    assert np.all(q[0] <= q[2])


@pytest.mark.parametrize("method", ["kmedoids", "forward", "random"])
def test_reduction_preserves_probability(method):
    V = rng.normal(size=(300, 12))
    idx, p = reduce_scenarios(V, 10, method=method, seed=0)
    assert len(idx) == 10
    assert abs(p.sum() - 1.0) < 1e-9


def test_forward_selection_beats_random_distance():
    from tavan.scenarios import kantorovich_distance
    V = rng.normal(size=(200, 8))
    idx_f, _ = fast_forward_selection(V, 15)
    idx_r, _ = reduce_scenarios(V, 15, "random", seed=1)
    df = kantorovich_distance(V, None, idx_f)
    dr = kantorovich_distance(V, None, idx_r)
    assert df <= dr


def test_risk_preserving_reduction_keeps_tail():
    # scenarios with a heavy upper tail in total
    base = rng.normal(0, 1, size=(400, 10))
    base[:20] += 8  # extreme scenarios
    s = ScenarioSet(base)
    full_cvar = s.cvar(0.95, loss="sum")
    red = s.reduce(30, method="cvar", loss="sum")
    red_cvar = red.cvar(0.95, loss="sum")
    plain = s.reduce(30, method="kmedoids")
    plain_cvar = plain.cvar(0.95, loss="sum")
    # tail-preserving reduction is closer to the true CVaR
    assert abs(red_cvar - full_cvar) <= abs(plain_cvar - full_cvar) + 1e-6


def test_energy_score_lower_for_matching():
    truth = rng.normal(size=(1, 20))
    good = rng.normal(loc=truth[0], scale=0.1, size=(200, 20))
    bad = rng.normal(loc=truth[0] + 5, scale=1.0, size=(200, 20))
    assert energy_score(truth, good) < energy_score(truth, bad)


def test_scenarioset_optimization_export():
    s = ScenarioSet(rng.normal(size=(10, 6, 2)), names=["a", "b"])
    opt = s.to_optimization()
    assert len(opt["scenarios"]) == 10
    assert abs(sum(opt["probability"].values()) - 1.0) < 1e-9
    assert (0, 0, "a") in opt["value"]


def test_json_roundtrip(tmp_path):
    s = ScenarioSet(rng.normal(size=(20, 8)), names=["x"], unit="MW")
    p = str(tmp_path / "s.json")
    s.to_json(p)
    s2 = ScenarioSet.from_json(p)
    assert np.allclose(s.values, s2.values)

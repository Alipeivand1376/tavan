"""Tests for the power module: DC flow, state estimation and FDI."""

import numpy as np
import pytest

from tavan.power import DCStateEstimator, FDIAttacker, ieee9, ieee14, ieee30, random_injection

rng = np.random.default_rng(0)


@pytest.mark.parametrize("case", [ieee9, ieee14, ieee30])
def test_dc_power_flow_balances(case):
    net = case()
    p = random_injection(net, seed=0)
    pf = net.dc_power_flow(p)
    recovered = net.incidence.T @ pf["flow"]
    assert np.allclose(recovered, p, atol=1e-6)


def test_dc_power_flow_batch():
    net = ieee14()
    P = random_injection(net, 10, seed=1)
    pf = net.dc_power_flow(P)
    assert pf["theta"].shape == (10, 14)
    assert pf["flow"].shape == (10, net.n_branch)


def test_state_estimation_recovers_angles():
    net = ieee14()
    est = DCStateEstimator(net, noise_std=0.1)
    p = random_injection(net, seed=2)
    truth = net.dc_power_flow(p)["theta"][net._noslack]
    z = est.simulate_measurements(p, seed=3)
    est_state = est.estimate(z).state
    assert np.sqrt(np.mean((est_state - truth) ** 2)) < 0.05


def test_clean_data_passes_detector():
    net = ieee14()
    est = DCStateEstimator(net, noise_std=0.5, alpha=0.01)
    alarms = 0
    for s in range(200):
        z = est.simulate_measurements(random_injection(net, seed=s), seed=1000 + s)
        alarms += int(est.detect(z)[0])
    assert alarms / 200 < 0.05  # false-alarm rate near alpha


def test_stealthy_attack_is_stealthy():
    net = ieee14()
    est = DCStateEstimator(net, noise_std=0.5)
    atk = FDIAttacker(est, knowledge=1.0, seed=3)
    caught = 0
    for s in range(100):
        clean = est.simulate_measurements(random_injection(net, seed=s), seed=5000 + s)
        a = atk.stealthy_attack(magnitude=0.2)
        caught += int(not atk.is_stealthy(clean, a))
    assert caught / 100 < 0.1  # stealthy attacks rarely detected


def test_stealthy_attack_shifts_state_without_residual():
    net = ieee14()
    est = DCStateEstimator(net, noise_std=0.3)
    atk = FDIAttacker(est, seed=1)
    clean = est.simulate_measurements(random_injection(net, seed=0), seed=1)
    a = atk.stealthy_attack(magnitude=0.3)
    r0 = est.estimate(clean)
    r1 = est.estimate(clean + a)
    assert np.abs(r1.state - r0.state).max() > 1e-3           # state moved
    assert abs(r1.objective - r0.objective) < 1e-6            # residual unchanged


def test_random_attack_is_detected():
    net = ieee14()
    est = DCStateEstimator(net, noise_std=0.3)
    atk = FDIAttacker(est, seed=2)
    caught = 0
    for s in range(50):
        clean = est.simulate_measurements(random_injection(net, seed=s), seed=7000 + s)
        caught += int(not atk.is_stealthy(clean, atk.random_attack(magnitude=1.0)))
    assert caught / 50 > 0.5


def test_mtd_detects_stale_attack():
    net = ieee14()
    est = DCStateEstimator(net, noise_std=0.4)
    atk = FDIAttacker(est, seed=4)
    P = random_injection(net, 100, seed=7)
    mtd = net.perturb_reactances(0.3, seed=11)
    est2 = DCStateEstimator(mtd, noise_std=0.4)
    dr = atk.evaluate_defense(P, est2, magnitude=0.25)
    assert dr["detection_rate"] > 0.5  # MTD catches most stale attacks


def test_fdi_dataset_labels():
    from tavan.recipes import fdi_dataset
    ds = fdi_dataset(ieee14(), n=200, attack_ratio=0.5, magnitude=0.3)
    assert ds.Z.shape[0] == 200
    assert 0.3 < ds.y.mean() < 0.7
    assert ds.delta[ds.y == 0].sum() == 0  # clean samples have no injection

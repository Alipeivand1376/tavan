"""Example 3 — False data injection vs moving-target defense.

Build a DC state estimator on the IEEE 14-bus system, show that a classic
stealthy FDI attack evades the chi-square bad-data detector, then show that a
moving-target-defense reactance perturbation exposes the now-stale attack.
Finally, build a labelled dataset and train a tiny detector with tavan's own
neural toolkit.

This is defensive security research on a simulated grid.

Run:  python examples/03_fdi_detection.py
"""

import numpy as np

import tavan
from tavan.power import DCStateEstimator, FDIAttacker, ieee14, random_injection
from tavan.recipes import fdi_dataset

net = ieee14()
est = DCStateEstimator(net, noise_std=0.5, alpha=0.01)
atk = FDIAttacker(est, knowledge=1.0, seed=0)

# 1. A stealthy attack a = H c shifts the state but not the residual
clean = est.simulate_measurements(random_injection(net, seed=1), seed=2)
a = atk.stealthy_attack(magnitude=0.25)
print("stealthy attack detected by chi-square test:", not atk.is_stealthy(clean, a))
print("  state shift:", round(np.abs(est.estimate(clean + a).state - est.estimate(clean).state).max(), 4), "rad")

# 2. Moving-target defense: perturb branch reactances (D-FACTS), re-estimate
mtd = net.perturb_reactances(factors=0.3, seed=7)
est_mtd = DCStateEstimator(mtd, noise_std=0.5)
P = random_injection(net, 200, seed=3)
dr = atk.evaluate_defense(P, est_mtd, magnitude=0.25)
print("MTD detection rate of the stale stealthy attack:", round(dr["detection_rate"], 3))

# 3. Labelled dataset + a small neural detector trained with tavan.nn
ds = fdi_dataset(net, n=1500, attack_ratio=0.5, magnitude=0.3, attack_type="partial")
(Xtr, ytr), (Xte, yte) = ds.split(test=0.3, seed=0)

from tavan.nn import MLP, Tensor, bce_with_logits, train
from tavan.config import TrainConfig

mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
clf = MLP([Xtr.shape[1], 64, 32, 1], activation="relu", rng=0)
train(clf, lambda m, xb, yb, **k: bce_with_logits(m(Tensor((xb - mu) / sd)), yb[:, None]),
      [Xtr, ytr.astype(float)], TrainConfig(epochs=150, lr=3e-3, patience=0))
pred = (clf(Tensor((Xte - mu) / sd)).data[:, 0] > 0).astype(int)
print("neural detector accuracy on held-out partial-knowledge attacks:",
      round(float((pred == yte).mean()), 3))

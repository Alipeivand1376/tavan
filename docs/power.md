# Power systems & false data injection

A small DC toolkit for cyber-physical studies on simulated grids.

```python
from tavan.power import ieee14, DCStateEstimator, FDIAttacker, random_injection

net = ieee14()
est = DCStateEstimator(net, noise_std=0.5)

# a stealthy attack a = H c evades the chi-square detector
atk = FDIAttacker(est)
clean = est.simulate_measurements(random_injection(net))
attack = atk.stealthy_attack(magnitude=0.25)
assert atk.is_stealthy(clean, attack)

# moving-target defense: perturb reactances, the stale attack is exposed
mtd = net.perturb_reactances(0.3)
est_mtd = DCStateEstimator(mtd, noise_std=0.5)
dr = atk.evaluate_defense(random_injection(net, 200), est_mtd)
print(dr["detection_rate"])
```

Build a labelled dataset for detector training:

```python
from tavan.recipes import fdi_dataset
ds = fdi_dataset(net, n=1500, attack_type="partial", magnitude=0.3)
(Xtr, ytr), (Xte, yte) = ds.split()
```

!!! note
    These tools are for defensive security research on simulated grids: training
    and evaluating detectors and defenses. The DC models are linear
    approximations; use MATPOWER/pandapower for AC studies.

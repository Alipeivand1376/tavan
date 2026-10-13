# Scenario generation

Fit a generative model to historical profiles and draw conditional,
physically-constrained scenarios.

```python
from tavan.generate import ScenarioGenerator, NonNegative, RampLimit

gen = ScenarioGenerator(
    model="diffusion",                     # bootstrap | copula | gmm | vae | diffusion | flow | gan
    conditions=["weekend", "doy"],         # calendar conditioning
    constraints=[NonNegative(), RampLimit(25.0)],
    model_params={"epochs": 300},
).fit(ts)

scen = gen.sample(500, date="2024-07-15")  # a ScenarioSet in physical units
```

## Working with a `ScenarioSet`

```python
scen.mean(); scen.quantiles([0.1, 0.5, 0.9])
scen.cvar(0.95, loss="max")                # peak-demand CVaR
small = scen.reduce(10, method="cvar", loss="max")   # tail-preserving reduction
scen.evaluate(reference_profiles)          # energy score, MMD, ...
scen.to_optimization()                     # dict for Pyomo/CVXPY
scen.report("report.html")                 # self-contained HTML
```

## Risk-aware sampling

Draw a tail-weighted, probability-weighted set so CVaR is estimated from many
more tail scenarios than plain Monte-Carlo:

```python
scen = gen.sample(200, date="2024-07-15", risk_alpha=0.95, risk_loss="max")
```

## Physical constraints

`NonNegative`, `Bounds`, `RampLimit`, `EnvelopeCap`, `EnergyBudget` and
`ClearSkyCap` (PV bounded by the clear-sky envelope, zero at night). Each can
both *project* scenarios and *measure* how often raw samples violate it.

## Drift detection

```python
result = gen.monitor(new_data)             # two-sample test + auto fine-tune
if result.drift: ...
```

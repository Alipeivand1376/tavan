"""Example 1 — Conditional, physically-constrained demand scenarios.

Fit a generative model to a demand series and draw day-ahead scenarios
conditioned on the day type, with non-negativity and ramp limits enforced.
Then reduce to a handful of representative scenarios for optimisation and
summarise the peak-demand risk (CVaR).

Run:  python examples/01_scenario_generation.py
"""

import tavan
from tavan.generate import NonNegative, RampLimit, ScenarioGenerator

# 1. Data — a year of hourly demand (replace with TimeSeries.from_csv(...))
ts = tavan.make_load(n_days=365, seed=0, missing_rate=0.02)
print(ts)

# 2. Fit a conditional generator (weekend + seasonal conditions)
gen = ScenarioGenerator(
    model="diffusion",                       # try "gmm" for an instant baseline
    conditions=["weekend", "doy"],
    constraints=[NonNegative(), RampLimit(max_up=25.0)],
    model_params={"epochs": 300},
    seed=0,
).fit(ts)

# 3. Sample 500 day-ahead scenarios for a specific date
scen = gen.sample(500, date="2024-07-15")
print(scen)
print("peak-demand CVaR 95%:", round(scen.cvar(0.95, loss="max"), 1), "MW")

# 4. Reduce to 10 representative scenarios (tail-preserving) for a UC/ED model
reduced = scen.reduce(10, method="cvar", loss="max")
print("reduced to", len(reduced), "scenarios; export:", list(reduced.to_optimization()))

# 5. One-call HTML report with fan chart, loss distribution and risk table
path = scen.report("demand_scenarios.html", reference=ts.to_profiles(), title="Day-ahead demand")
print("report written to", path)

"""
tavan — quickstart demo
=======================

Run after:  pip install "tavan[plot]"
Then:       python demo_quickstart.py

Shows the four core capabilities with printed metrics and plots.
Each section is independent — comment any out to focus on one.
"""

import matplotlib.pyplot as plt
import numpy as np

import tavan
from tavan.forecast import ConformalForecaster, DLinear, NHiTS, SeasonalNaive, backtest, coverage
from tavan.generate import NonNegative, RampLimit, ScenarioGenerator
from tavan.power import DCStateEstimator, FDIAttacker, ieee14, random_injection


# ======================================================================
# 1. SCENARIO GENERATION  — day-ahead demand scenarios with risk analysis
# ======================================================================
def demo_scenarios():
    print("\n" + "=" * 60 + "\n1) SCENARIO GENERATION\n" + "=" * 60)
    ts = tavan.make_load(n_days=365, seed=0, missing_rate=0.02)   # 1 year, 2% missing
    print(ts)

    gen = ScenarioGenerator(
        model="gmm",                       # fast; try "diffusion" for the deep model
        conditions=["weekend", "doy"],     # condition on day type & season
        constraints=[NonNegative(), RampLimit(max_up=25.0)],
        seed=0,
    ).fit(ts)

    scen = gen.sample(500, date="2024-07-15")                    # 500 scenarios for one day
    print(f"generated {scen.n_scenarios} scenarios, horizon {scen.horizon}")
    print(f"peak-demand VaR 95%:  {scen.var(0.95, loss='max'):.1f} MW")
    print(f"peak-demand CVaR 95%: {scen.cvar(0.95, loss='max'):.1f} MW")

    reduced = scen.reduce(10, method="cvar", loss="max")         # 10 representatives for optimisation
    print(f"reduced to {len(reduced)} scenarios (Kantorovich dist "
          f"{reduced.meta['reduction_distance']:.2f})")

    scen.report("demand_scenarios.html", title="Day-ahead demand")
    print("-> wrote demand_scenarios.html (open it in a browser)")

    scen.plot()                                                 # fan chart
    plt.tight_layout()


# ======================================================================
# 2. FORECASTING  — compare models + calibrated prediction intervals
# ======================================================================
def demo_forecasting():
    print("\n" + "=" * 60 + "\n2) FORECASTING\n" + "=" * 60)
    y = tavan.make_load(n_days=150, seed=0).to_numpy()
    H, L = 24, 96

    print("walk-forward backtest (skill vs seasonal-naive):")
    for name, model, folds in [
        ("seasonal-naive", SeasonalNaive(24, H), 5),
        ("DLinear",        DLinear(L, H),        5),
        ("N-HiTS",         NHiTS(L, H, epochs=80), 3),
    ]:
        r = backtest(model, y, horizon=H, n_folds=folds, season_length=24)
        print(f"  {name:15s} MAE={r.metrics['mae']:6.3f}  skill={r.skill['mae']:+.1%}")

    # calibrated 90% interval with conformal prediction (distribution-free)
    cf = ConformalForecaster(DLinear(L, H), levels=(0.9,)).fit(y[:-H])
    fc = cf.predict()
    lo, hi = fc.interval(0.9)
    print(f"last-day 90% interval coverage: {coverage(y[-H:], lo, hi):.2f}")

    ax = fc.plot(history=y[-96:])
    ax.set_title("DLinear forecast with 90% conformal interval")
    plt.tight_layout()


# ======================================================================
# 3. POWER / CYBERSECURITY  — stealthy FDI attack vs moving-target defense
# ======================================================================
def demo_fdi():
    print("\n" + "=" * 60 + "\n3) FDI ATTACK vs MOVING-TARGET DEFENSE\n" + "=" * 60)
    net = ieee14()
    est = DCStateEstimator(net, noise_std=0.5, alpha=0.01)
    atk = FDIAttacker(est, knowledge=1.0, seed=0)

    clean = est.simulate_measurements(random_injection(net, seed=1), seed=2)
    attack = atk.stealthy_attack(magnitude=0.25)                # a = H c
    print(f"stealthy attack caught by chi-square detector: {not atk.is_stealthy(clean, attack)}")
    shift = np.abs(est.estimate(clean + attack).state - est.estimate(clean).state).max()
    print(f"  it still shifts the state estimate by {shift:.3f} rad (residual unchanged)")

    mtd = net.perturb_reactances(factors=0.3, seed=7)           # D-FACTS reactance perturbation
    est_mtd = DCStateEstimator(mtd, noise_std=0.5)
    dr = atk.evaluate_defense(random_injection(net, 200, seed=3), est_mtd, magnitude=0.25)
    print(f"after MTD, detection rate of the now-stale attack: {dr['detection_rate']:.1%}")


# ======================================================================
# 4. PREPROCESSING  — seasonal imputation recovers a long gap
# ======================================================================
def demo_imputation():
    print("\n" + "=" * 60 + "\n4) MISSING-DATA IMPUTATION\n" + "=" * 60)
    clean = tavan.make_load(n_days=10, seed=3).to_numpy()
    corrupt = clean.copy()
    corrupt[70:82] = np.nan                                      # knock out a 12-hour block

    filled = tavan.datasets.TimeSeries(corrupt, start="2024-01-01", freq="1h").impute("seasonal").to_numpy()
    err = np.sqrt(np.mean((filled[70:82] - clean[70:82]) ** 2))
    print(f"seasonal imputation RMSE on the gap: {err:.2f} MW")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(clean, color="lightgray", lw=3, label="truth")
    ax.plot(filled, color="#08519c", lw=1, label="imputed")
    ax.axvspan(70, 82, color="red", alpha=0.1, label="missing block")
    ax.legend(); ax.set_title("Seasonal imputation of a 12-hour gap")
    plt.tight_layout()


if __name__ == "__main__":
    demo_scenarios()
    demo_forecasting()
    demo_fdi()
    demo_imputation()
    print("\nAll demos done. Showing plots...")
    plt.show()          # opens the figure windows

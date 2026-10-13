# tavan

**Lightweight time-series scenario generation and forecasting for power systems, energy and finance.**

`tavan` (توان — Persian for *power / energy*) is a research-grade Python library whose **only hard dependency is NumPy**. Every deep-learning model — diffusion, VAE, normalising flow, GAN, and the N-HiTS / TiDE / TSMixer / PatchTST forecasters — runs on a compact **built-in reverse-mode automatic-differentiation engine** written from scratch on NumPy, on CPU, with no PyTorch or TensorFlow. The autograd core is gradient-checked against finite differences, and the whole library ships with a 100+ test suite.

The design goal is a small, transparent, extensible toolkit where the interesting algorithms are implemented in the library itself rather than imported from heavy frameworks — fast to install, easy to read, and genuinely novel in the parts that matter.

```python
import tavan

ts   = tavan.make_load(120)                                    # demo demand series
gen  = tavan.ScenarioGenerator("diffusion",                    # fit a generative model
                               conditions=["weekend", "doy"]).fit(ts)
scen = gen.sample(500, date="2024-07-15")                      # 500 day-ahead scenarios
print(scen.cvar(0.95, loss="max"))                             # peak-demand CVaR
scen.report("report.html")                                     # self-contained HTML report

fc = tavan.Forecaster("dlinear", intervals=(0.9,)).fit(ts).predict()   # interval forecast
lo, hi = fc.interval(0.9)
```

---

## Why another library?

Existing tools (SDV, Darts, sktime, GluonTS) are excellent but heavy and general. `tavan` is deliberately different:

- **No deep-learning framework.** A ~600-line vectorised autograd engine (`tavan.nn`) powers every neural model. Installation is `pip install tavan` and nothing else. This is the core novelty — the models are *implemented*, not wrapped.
- **Power-systems aware.** Physics-aware constraints (clear-sky PV envelopes, ramp limits, non-negativity, energy budgets), a DC power-flow / state-estimation / false-data-injection toolkit, and moving-target-defense experiments are built in.
- **Risk-first.** CVaR-aware scenario sampling, tail-preserving scenario reduction, and direct export to stochastic-optimisation models.
- **Honest evaluation.** Walk-forward backtesting always reports skill against the seasonal-naive baseline; conformal prediction gives distribution-free interval guarantees.
- **Extensible.** Every model is registered under a name and discoverable; third-party packages can add models through entry points without touching `tavan`.

## Installation

```bash
pip install tavan                 # core (NumPy only)
pip install "tavan[plot]"         # + matplotlib for plotting helpers
pip install "tavan[plot,pandas]"  # + pandas interop
```

Python ≥ 3.9.

## Capabilities

### 1 · Scenario generation (`tavan.generate`)

Fit a generative model to historical profiles and draw conditional, physically-constrained scenarios.

| Model | `name` | Notes |
|-------|--------|-------|
| Analog / bootstrap | `bootstrap` | instant baseline, analog ensembles |
| Gaussian copula | `copula` | empirical marginals + correlation, millisecond fit |
| Gaussian mixture (EM) | `gmm` | multi-modal, BIC model selection, exact likelihoods |
| Variational auto-encoder | `vae` | conditional, KL-annealed |
| Denoising diffusion | `diffusion` | v-prediction + DDIM, classifier-free guidance |
| Normalising flow (RealNVP) | `flow` | **exact** likelihoods → anomaly detection |
| Conditional GAN | `gan` | moment-matching for small-data stability |

High-level `ScenarioGenerator` wraps cleaning → imputation → scaling → conditioning → physical constraints → CVaR-aware sampling. Scenario sets support quantiles, VaR/CVaR, reduction (`kmedoids`, fast-forward selection, tail-preserving), quality metrics (energy score, MMD, variogram score, a classifier two-sample test), drift detection, and one-call HTML reports.

### 2 · Forecasting (`tavan.forecast`)

Modern, **non-recurrent** point and probabilistic forecasters — no RNNs.

- **Baselines:** `naive`, `seasonal_naive`, `drift`, `mean`
- **Linear (closed-form ridge, millisecond training):** `dlinear`, `nlinear`
- **Reservoir computing:** `ngrc` — next-generation reservoir / nonlinear vector autoregression
- **Neural (shared autograd):** `nhits`, `tide`, `tsmixer`, `patchtst`

All support multivariate series, direct multi-step horizons, exogenous/calendar covariates and quantile (pinball) outputs. `ConformalForecaster` adds calibrated prediction intervals to any point model; `backtest` does walk-forward evaluation with seasonal-naive skill.

### 3 · Power systems (`tavan.power`)

DC power flow, weighted-least-squares state estimation with chi-square bad-data detection, stealthy/partial/random **false-data-injection** attack generation, labelled detector-training datasets, and **moving-target-defense** evaluation — on bundled IEEE 9/14/30-bus cases. For defensive security research on simulated grids.

## Examples

The [`examples/`](examples) directory contains runnable scripts:

1. [`01_scenario_generation.py`](examples/01_scenario_generation.py) — conditional, constrained demand scenarios → reduction → CVaR → HTML report
2. [`02_forecasting.py`](examples/02_forecasting.py) — backtest several models vs seasonal-naive, conformal intervals
3. [`03_fdi_detection.py`](examples/03_fdi_detection.py) — stealthy FDI evades detection, MTD exposes it, train a neural detector

## A note on scope and honesty

`tavan` is a research and prototyping tool. The neural models run on CPU and are intended for the dataset sizes common in power systems (hundreds to a few thousand daily profiles), not for million-sample industrial training — for that, use a GPU framework. The clear-sky and DC-power-flow models are deliberately simple physical approximations; for bankable studies use dedicated tools (pvlib, pandapower, MATPOWER). What `tavan` offers is a transparent, dependency-light, end-to-end pipeline you can read, extend and cite.

## Extending

```python
from tavan.generate import BaseGenerator
from tavan.registry import register

@register("generator", "my_model")
class MyModel(BaseGenerator):
    def _fit(self, X, cond): ...
    def _sample(self, n, cond, rng): ...

tavan.ScenarioGenerator("my_model").fit(data)   # usable everywhere by name
```

Third-party packages can also expose models through the `tavan.plugins` entry-point group.

## Development

```bash
git clone https://github.com/Alipeivand1376/tavan
cd tavan
pip install -e ".[dev]"
pytest          # 100+ tests, including finite-difference gradient checks
ruff check src
```

## License

MIT © Ali Peivand

## Citation

If you use `tavan` in academic work, please cite it (see [`CITATION.cff`](CITATION.cff)).

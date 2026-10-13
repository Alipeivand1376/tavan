---
title: 'tavan: A dependency-light Python library for time-series scenario generation and forecasting in power systems'
tags:
  - Python
  - time series
  - scenario generation
  - probabilistic forecasting
  - power systems
  - generative models
  - conformal prediction
authors:
  - name: Ali Peivand
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 13 October 2026
bibliography: paper.bib
---

# Summary

`tavan` is a Python library for probabilistic time-series work — scenario
generation and forecasting — aimed at power systems, energy and finance. Its
distinguishing feature is that its only hard dependency is NumPy: the
generative models (denoising diffusion, variational auto-encoder, normalising
flow, conditional GAN, Gaussian mixture, copula) and the neural forecasters
(N-HiTS, TiDE, TSMixer, PatchTST) are all implemented on a compact,
gradient-checked reverse-mode automatic-differentiation engine written on
NumPy and running on CPU. This makes `tavan` trivial to install and deploy,
easy to read and audit, and well suited to the moderate dataset sizes typical
of power-systems research, where the number of daily profiles is in the
hundreds to low thousands.

Around these models `tavan` provides a complete, power-systems-aware workflow:
missing-data imputation (including seasonal, k-nearest-neighbour and
expectation–maximisation methods), physics-aware scenario constraints
(clear-sky photovoltaic envelopes, ramp limits, energy budgets),
risk-aware (CVaR) scenario sampling and tail-preserving scenario reduction for
stochastic optimisation, proper scoring-rule evaluation, conformal prediction
intervals, honest walk-forward backtesting, and a small DC power-flow, state
estimation and false-data-injection toolkit with moving-target-defense
experiments.

# Statement of need

Probabilistic scenarios and forecasts are central to modern power-system
operation and planning under uncertainty from renewables, demand and
adversarial manipulation. Existing general-purpose libraries for synthetic
data (SDV) and forecasting (Darts, GluonTS, sktime) are powerful but depend on
large deep-learning frameworks and are not specialised for the physical
constraints, risk measures and cyber-physical security concerns of power
systems. Researchers who need a transparent, lightweight tool — one they can
install without a GPU stack, read in full, extend, and cite — are not well
served.

`tavan` fills this gap. By implementing its models on a self-contained NumPy
autograd engine it removes the framework dependency entirely while keeping the
algorithms modern and inspectable. It couples generation and forecasting to the
domain: generated scenarios can be bounded by physical envelopes and exported
directly to stochastic-optimisation models; forecasts are always benchmarked
against the seasonal-naive baseline and can be wrapped in distribution-free
conformal intervals; and a built-in state-estimation and false-data-injection
module lets researchers reproduce the classic stealthy-attack result and
evaluate moving-target defenses on standard IEEE test systems. Every component
is registered under a name and discoverable, and third-party models can be
contributed through an entry-point plugin mechanism without modifying `tavan`.

# Design and functionality

The core (`tavan.nn`) is a roughly 600-line reverse-mode autograd engine with
layers, optimisers (Adam/AdamW, SGD), schedules and a training loop with early
stopping; its gradients are verified against finite differences in the test
suite. On top of it, `tavan.generate` provides seven generators behind a common
`fit`/`sample` interface and a high-level `ScenarioGenerator` that handles
preprocessing, calendar conditioning and physical constraints;
`tavan.scenarios` provides a weighted `ScenarioSet` with VaR/CVaR, reduction
(k-medoids, fast-forward selection, risk-preserving) and quality metrics
(energy score, variogram score, maximum mean discrepancy, a classifier
two-sample test). `tavan.forecast` provides baseline, linear (DLinear/NLinear,
fitted in closed form), reservoir-computing (NG-RC) and neural forecasters,
with conformal intervals and walk-forward backtesting. `tavan.power` provides DC
power flow, weighted-least-squares state estimation with chi-square bad-data
detection, and false-data-injection attack and defense tooling on bundled IEEE
9/14/30-bus cases.

# Acknowledgements

The author thanks the open-source scientific-Python community.

# References

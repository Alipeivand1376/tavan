# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-13

Initial release.

### Added
- `tavan.nn`: NumPy reverse-mode autograd engine, layers, optimisers, training loop.
- `tavan.data`: `TimeSeries` container, windowing, offline demo datasets (load, PV, wind, price).
- `tavan.preprocess`: imputation (linear, seasonal, kNN, EM), outlier cleaning, scalers, calendar features, pipelines.
- `tavan.generate`: bootstrap, Gaussian copula, Gaussian mixture, VAE, diffusion, RealNVP flow, conditional GAN; `ScenarioGenerator` pipeline; physics-aware constraints; drift detection.
- `tavan.scenarios`: `ScenarioSet`, VaR/CVaR, reduction (k-medoids, fast-forward, risk-preserving), quality metrics.
- `tavan.forecast`: baselines, DLinear/NLinear, NG-RC, N-HiTS, TiDE, TSMixer, PatchTST; conformal intervals; walk-forward backtesting.
- `tavan.power`: DC power flow, state estimation, false-data-injection tooling, IEEE 9/14/30-bus cases.
- `tavan.recipes`: one-call domain entry points.
- Self-contained HTML reports; optional matplotlib plotting.
- 100+ tests including finite-difference gradient checks.

"""Domain recipes — opinionated one-call entry points.

These wrap the generic machinery with sensible defaults for recurring
power-system and finance tasks, so common jobs are a single call while every
underlying object stays accessible for customisation.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .data.timeseries import as_timeseries
from .generate import ClearSkyCap, NonNegative, RampLimit, ScenarioGenerator
from .scenarios import ScenarioSet


def load_scenarios(data, n: int = 200, model: str = "gmm", date=None, weekend: Sequence[int] = (5, 6),
                   conditions=("weekend", "doy"), ramp_limit: Optional[float] = None,
                   risk_alpha: Optional[float] = None, **kwargs) -> ScenarioSet:
    """Electricity-demand scenarios.

    Non-negativity is enforced; pass ``ramp_limit`` (MW/step) to cap ramps
    and ``risk_alpha`` (e.g. 0.95) for a CVaR-focused, tail-weighted set.
    Peak demand is a natural loss, so use ``scen.cvar(loss="max")``.
    """
    cons = [NonNegative()]
    if ramp_limit:
        cons.append(RampLimit(ramp_limit))
    gen = ScenarioGenerator(model, conditions=conditions, weekend=weekend, constraints=cons,
                            model_params=kwargs.pop("model_params", None), **kwargs)
    gen.fit(data)
    return gen.sample(n, date=date, risk_alpha=risk_alpha, risk_loss="max")


def pv_scenarios(data, latitude: float, longitude: float, tz_offset: float, n: int = 200,
                 capacity: Optional[float] = None, model: str = "diffusion", date=None,
                 conditions=("doy",), **kwargs) -> ScenarioSet:
    """Photovoltaic generation scenarios bounded by the clear-sky envelope.

    The :class:`ClearSkyCap` constraint guarantees zero output at night and
    physically plausible daytime values.  ``capacity`` defaults to the
    observed maximum.
    """
    ts = as_timeseries(data)
    cap = capacity or float(np.nanmax(ts.values))
    freq_min = None if ts.freq is None else float(ts.freq / np.timedelta64(1, "m"))
    cons = [NonNegative(), ClearSkyCap(latitude, longitude, cap, tz_offset, freq_minutes=freq_min)]
    gen = ScenarioGenerator(model, conditions=conditions, constraints=cons,
                            model_params=kwargs.pop("model_params", None), **kwargs)
    gen.fit(ts)
    return gen.sample(n, date=date)


def price_scenarios(data, n: int = 500, model: str = "flow", horizon_conditions=None,
                    risk_alpha: Optional[float] = 0.95, **kwargs) -> ScenarioSet:
    """Price / returns scenarios for market and risk analysis.

    Uses a normalising flow by default (heavy tails, exact likelihood) and
    returns a tail-weighted set for Value-at-Risk / Expected-Shortfall work.
    """
    gen = ScenarioGenerator(model, conditions=None, scale="standard",
                            model_params=kwargs.pop("model_params", None), **kwargs)
    gen.fit(data)
    return gen.sample(n, risk_alpha=risk_alpha, risk_loss="neg_min")


def fdi_dataset(network, injections=None, n: int = 1000, attack_ratio: float = 0.5, magnitude: float = 0.2,
                attack_type: str = "stealthy", knowledge: float = 1.0, noise_std: float = 1.0,
                seed: Optional[int] = 0):
    """Labelled false-data-injection dataset for detector training.

    Builds a DC estimator on ``network`` and generates attacked/clean
    measurement vectors from random (or supplied) injection scenarios.
    Returns an :class:`~tavan.power.AttackDataset`.
    """
    from .power import DCStateEstimator, FDIAttacker, random_injection

    est = DCStateEstimator(network, noise_std=noise_std)
    atk = FDIAttacker(est, knowledge=knowledge, seed=seed)
    if injections is None:
        injections = random_injection(network, n, seed=seed)
    return atk.make_dataset(injections, attack_ratio=attack_ratio, magnitude=magnitude, attack_type=attack_type)


def stress_test(generator_or_scenarios, loss="max", alphas=(0.9, 0.95, 0.99), n: int = 2000,
                date=None) -> dict:
    """Summarise tail risk of a fitted generator or an existing scenario set."""
    if isinstance(generator_or_scenarios, ScenarioSet):
        scen = generator_or_scenarios
    else:
        scen = generator_or_scenarios.sample(n, date=date)
    return scen.risk_summary(loss=loss, alphas=alphas)

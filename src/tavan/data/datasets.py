"""Synthetic but realistic demo datasets (generated, no downloads).

They make every example and test reproducible offline.  Each function
returns a :class:`~tavan.data.TimeSeries` with timestamps.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .._stats import norm_cdf
from .._validation import make_rng
from .timeseries import TimeSeries
from .timeutils import date_range, day_of_week, day_of_year, hour_of_day, parse_freq, steps_per


def _ar1(n: int, phi: float, sigma: float, rng) -> np.ndarray:
    e = rng.normal(0, sigma, n)
    out = np.empty(n)
    out[0] = e[0] / np.sqrt(max(1 - phi ** 2, 1e-6))
    for i in range(1, n):
        out[i] = phi * out[i - 1] + e[i]
    return out


def _corrupt(values: np.ndarray, missing_rate: float, outlier_rate: float, rng,
             gap_length: int = 6) -> np.ndarray:
    v = values.copy()
    T = v.shape[0]
    if outlier_rate > 0:
        k = int(outlier_rate * T)
        idx = rng.choice(T, size=k, replace=False)
        v[idx, 0] = v[idx, 0] * rng.choice([0.2, 2.5, 3.0], size=k)
    if missing_rate > 0:
        n_gaps = max(1, int(missing_rate * T / gap_length))
        for s in rng.integers(0, T - gap_length, size=n_gaps):
            v[s:s + rng.integers(1, 2 * gap_length), 0] = np.nan
    return v


def make_load(n_days: int = 365, freq: str = "1h", start: str = "2024-01-01", peak: float = 100.0,
              with_temperature: bool = False, missing_rate: float = 0.0, outlier_rate: float = 0.0,
              seed: Optional[int] = 0) -> TimeSeries:
    """Electricity demand with daily double peak, weekly and annual cycles.

    Parameters
    ----------
    n_days, freq, start:
        Length, resolution and first timestamp.
    peak:
        Approximate annual peak (MW).
    with_temperature:
        Add an ambient-temperature channel (°C) that drives the demand.
    missing_rate, outlier_rate:
        Fraction of corrupted samples (for preprocessing demos).
    """
    rng = make_rng(seed)
    spd = steps_per(freq, "1D")
    n = n_days * spd
    ts = date_range(start, n, freq)
    h = hour_of_day(ts)
    doy = day_of_year(ts)
    dow = day_of_week(ts)
    # temperature: annual cycle + diurnal cycle + weather noise
    temp = 18 - 10 * np.cos(2 * np.pi * (doy - 15) / 365.25) + 5 * np.sin(2 * np.pi * (h - 9) / 24) \
        + _ar1(n, 0.995 ** (24 / spd), 0.35, rng)
    daily = (0.62 + 0.22 * np.exp(-0.5 * ((h - 8.5) / 1.8) ** 2)
             + 0.33 * np.exp(-0.5 * ((h - 19.5) / 2.2) ** 2)
             - 0.08 * np.exp(-0.5 * ((h - 3.5) / 2.0) ** 2))
    weekly = np.where(dow == 5, 0.90, np.where(dow == 6, 0.84, 1.0))
    weather = 1 + 0.012 * np.maximum(temp - 22, 0) + 0.008 * np.maximum(14 - temp, 0)
    noise = 1 + _ar1(n, 0.9, 0.012, rng)
    load = peak / 1.25 * daily * weekly * weather * noise
    values = np.column_stack([load, temp]) if with_temperature else load[:, None]
    values = _corrupt(values, missing_rate, outlier_rate, rng, gap_length=max(2, spd // 4))
    names = ["load", "temperature"] if with_temperature else ["load"]
    return TimeSeries(values, timestamps=ts, freq=freq, names=names, unit="MW")


def make_pv(n_days: int = 365, freq: str = "1h", start: str = "2024-01-01", capacity: float = 1.0,
            latitude: float = 35.7, longitude: float = 51.4, tz_offset: float = 3.5,
            missing_rate: float = 0.0, seed: Optional[int] = 0) -> TimeSeries:
    """Photovoltaic output = clear-sky envelope × stochastic cloud index.

    Days follow a three-state Markov chain (clear / partly cloudy /
    overcast) with intra-day variability.  Defaults are located in Tehran.
    """
    from ..power.solar import clear_sky_pv

    rng = make_rng(seed)
    spd = steps_per(freq, "1D")
    n = n_days * spd
    ts = date_range(start, n, freq)
    minutes = parse_freq(freq) / np.timedelta64(1, "m")
    env = clear_sky_pv(ts, latitude, longitude, capacity, tz_offset, freq_minutes=minutes)
    P = np.array([[0.75, 0.2, 0.05], [0.35, 0.45, 0.2], [0.2, 0.4, 0.4]])
    state = 0
    kt = np.empty(n)
    means, sds = (0.95, 0.7, 0.3), (0.02, 0.15, 0.12)
    for d in range(n_days):
        state = rng.choice(3, p=P[state])
        intra = _ar1(spd, 0.7, sds[state], rng)
        kt[d * spd:(d + 1) * spd] = np.clip(means[state] + intra, 0.03, 1.05)
    pv = np.clip(env * kt, 0, capacity)
    values = _corrupt(pv[:, None], missing_rate, 0.0, rng, gap_length=max(2, spd // 6))
    return TimeSeries(values, timestamps=ts, freq=freq, names=["pv"], unit="MW")


def make_wind(n_days: int = 365, freq: str = "1h", start: str = "2024-01-01", capacity: float = 2.0,
              mean_speed: float = 7.5, seed: Optional[int] = 0) -> TimeSeries:
    """Wind-farm output from a Weibull-distributed, autocorrelated speed."""
    rng = make_rng(seed)
    spd = steps_per(freq, "1D")
    n = n_days * spd
    ts = date_range(start, n, freq)
    z = _ar1(n, 0.97 ** (24 / spd), 1.0, rng)
    z = z / z.std()
    u = norm_cdf(z)
    k_shape, scale = 2.0, mean_speed / 0.886
    speed = scale * (-np.log(1 - np.clip(u, 1e-9, 1 - 1e-9))) ** (1 / k_shape)
    cut_in, rated, cut_out = 3.0, 12.0, 25.0
    p = np.where(speed < cut_in, 0.0,
                 np.where(speed < rated, ((speed - cut_in) / (rated - cut_in)) ** 3,
                          np.where(speed < cut_out, 1.0, 0.0)))
    return TimeSeries(capacity * p[:, None], timestamps=ts, freq=freq, names=["wind"], unit="MW")


def make_price(n_steps: int = 1000, freq: str = "1D", start: str = "2022-01-03", s0: float = 100.0,
               drift: float = 0.05, vol: float = 0.2, jump_rate: float = 0.02, seed: Optional[int] = 0) -> TimeSeries:
    """Asset price with stochastic volatility and jumps (finance demos).

    Geometric Brownian motion whose volatility follows a mean-reverting
    log-AR(1) process, plus Poisson jumps.  ``drift`` and ``vol`` are annual.
    """
    rng = make_rng(seed)
    dt = 1 / 252
    logv = np.log(vol) + _ar1(n_steps, 0.98, 0.08, rng)
    sig = np.exp(logv)
    jumps = rng.poisson(jump_rate, n_steps) * rng.normal(-0.01, 0.04, n_steps)
    r = (drift - 0.5 * sig ** 2) * dt + sig * np.sqrt(dt) * rng.normal(size=n_steps) + jumps
    price = s0 * np.exp(np.cumsum(r))
    ts = date_range(start, n_steps, freq)
    return TimeSeries(price[:, None], timestamps=ts, freq=freq, names=["price"], unit="USD")

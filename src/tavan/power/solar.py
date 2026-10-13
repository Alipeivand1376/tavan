"""Solar geometry and clear-sky irradiance (no external dependencies).

The models are deliberately simple and well documented so that they can
serve as *physical envelopes* for generated PV scenarios:

* solar position from Cooper's declination and Spencer-style equation of
  time;
* clear-sky global horizontal irradiance (GHI) from the Haurwitz (1945)
  model, ``GHI = 1098 · cos(z) · exp(-0.059 / cos(z))``.

For bankable irradiance studies use dedicated tools such as *pvlib*.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..data.timeutils import day_of_year, hour_of_day, to_datetime64


def declination(doy) -> np.ndarray:
    """Solar declination in radians (Cooper, 1969)."""
    doy = np.asarray(doy, dtype=float)
    return np.deg2rad(23.45) * np.sin(2 * np.pi * (284 + doy) / 365.0)


def equation_of_time(doy) -> np.ndarray:
    """Equation of time in minutes."""
    b = 2 * np.pi * (np.asarray(doy, dtype=float) - 81) / 364.0
    return 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)


def solar_zenith(timestamps, latitude: float, longitude: float, tz_offset: float = 0.0,
                 shift_minutes: float = 0.0) -> np.ndarray:
    """Solar zenith angle (radians).

    Parameters
    ----------
    timestamps:
        Local standard-time timestamps.
    latitude, longitude:
        Degrees (east positive).
    tz_offset:
        Hours between local standard time and UTC (e.g. 3.5 for Tehran).
    shift_minutes:
        Evaluate the position this many minutes after each stamp (use half
        the sampling interval for interval-averaged data).
    """
    ts = to_datetime64(timestamps)
    doy = day_of_year(ts)
    hours = hour_of_day(ts) + shift_minutes / 60.0
    solar_time = hours + (4.0 * (longitude - 15.0 * tz_offset) + equation_of_time(doy)) / 60.0
    omega = np.deg2rad(15.0 * (solar_time - 12.0))
    phi = np.deg2rad(latitude)
    dec = declination(doy)
    cosz = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(omega)
    return np.arccos(np.clip(cosz, -1.0, 1.0))


def clear_sky_ghi(timestamps, latitude: float, longitude: float, tz_offset: float = 0.0,
                  shift_minutes: float = 0.0) -> np.ndarray:
    """Clear-sky GHI in W/m² (Haurwitz model)."""
    cosz = np.cos(solar_zenith(timestamps, latitude, longitude, tz_offset, shift_minutes))
    ghi = np.zeros_like(cosz)
    up = cosz > 1e-3
    ghi[up] = 1098.0 * cosz[up] * np.exp(-0.059 / cosz[up])
    return ghi


def clear_sky_pv(timestamps, latitude: float, longitude: float, capacity: float = 1.0,
                 tz_offset: float = 0.0, derate: float = 0.85, freq_minutes: Optional[float] = None) -> np.ndarray:
    """Clear-sky PV output envelope.

    ``P = capacity · derate · GHI / 1000`` capped at ``capacity``.  When
    ``freq_minutes`` is given, the sun position is evaluated at mid-interval.
    """
    shift = 0.5 * freq_minutes if freq_minutes else 0.0
    ghi = clear_sky_ghi(timestamps, latitude, longitude, tz_offset, shift)
    return np.minimum(capacity * derate * ghi / 1000.0, capacity)


def daylight_mask(timestamps, latitude: float, longitude: float, tz_offset: float = 0.0,
                  shift_minutes: float = 0.0) -> np.ndarray:
    """``True`` when the sun is above the horizon."""
    return solar_zenith(timestamps, latitude, longitude, tz_offset, shift_minutes) < np.pi / 2

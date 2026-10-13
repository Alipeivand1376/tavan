"""Small statistical special functions implemented without SciPy."""

from __future__ import annotations

import math

import numpy as np

_erf_u = np.frompyfunc(math.erf, 1, 1)
_erfc_u = np.frompyfunc(math.erfc, 1, 1)


def erf(x) -> np.ndarray:
    return np.asarray(_erf_u(np.asarray(x, dtype=float)), dtype=float)


def norm_cdf(x) -> np.ndarray:
    """Standard normal CDF."""
    x = np.asarray(x, dtype=float)
    return 0.5 * np.asarray(_erfc_u(-x / math.sqrt(2.0)), dtype=float)


def norm_pdf(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


# Acklam's rational approximation coefficients
_A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
_B = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01]
_C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
_D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]


def norm_ppf(p) -> np.ndarray:
    """Inverse standard normal CDF (Acklam + one Halley refinement step)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-300, 1 - 1e-16)
    x = np.empty_like(p)
    lo, hi = 0.02425, 1 - 0.02425
    m = p < lo
    if m.any():
        q = np.sqrt(-2 * np.log(p[m]))
        x[m] = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
               ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    m = (p >= lo) & (p <= hi)
    if m.any():
        q = p[m] - 0.5
        r = q * q
        x[m] = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
               (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1)
    m = p > hi
    if m.any():
        q = np.sqrt(-2 * np.log(1 - p[m]))
        x[m] = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
                ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    e = norm_cdf(x) - p
    u = e * math.sqrt(2 * math.pi) * np.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def chi2_ppf(p: float, dof: float) -> float:
    """Chi-square quantile (Wilson–Hilferty approximation, accurate for dof ≥ 3)."""
    z = float(norm_ppf(np.array([p]))[0])
    h = 2.0 / (9.0 * dof)
    return float(dof * (1 - h + z * math.sqrt(h)) ** 3)


def chi2_sf(x, dof: float) -> np.ndarray:
    """Chi-square survival function via Wilson–Hilferty."""
    x = np.maximum(np.asarray(x, dtype=float), 1e-300)
    h = 2.0 / (9.0 * dof)
    z = ((x / dof) ** (1 / 3) - (1 - h)) / math.sqrt(h)
    return 1 - norm_cdf(z)


def ks_2samp(a, b) -> tuple:
    """Two-sample Kolmogorov–Smirnov statistic and asymptotic p-value."""
    a = np.sort(np.asarray(a, dtype=float).ravel())
    b = np.sort(np.asarray(b, dtype=float).ravel())
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side="right") / a.size
    cdf_b = np.searchsorted(b, allv, side="right") / b.size
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    en = math.sqrt(a.size * b.size / (a.size + b.size))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 1e-3:
        return d, 1.0
    p = 2 * sum((-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam) for j in range(1, 101))
    return d, float(min(max(p, 0.0), 1.0))


def weighted_quantile(x, q, weights=None, axis: int = 0) -> np.ndarray:
    """Quantiles of ``x`` along ``axis`` with optional sample weights."""
    x = np.asarray(x, dtype=float)
    qs = np.atleast_1d(np.asarray(q, dtype=float))
    if weights is None:
        out = np.quantile(x, qs, axis=axis)
        return out if np.ndim(q) else out[0]
    w = np.asarray(weights, dtype=float)
    xm = np.moveaxis(x, axis, 0)
    order = np.argsort(xm, axis=0)
    xs = np.take_along_axis(xm, order, axis=0)
    ws = w[order] if xm.ndim > 1 else w[order]
    cw = np.cumsum(ws, axis=0)
    cw = cw / cw[-1:]
    res = []
    for qq in qs:
        idx = np.argmax(cw >= qq - 1e-12, axis=0)
        res.append(np.take_along_axis(xs, np.expand_dims(idx, 0), axis=0)[0])
    out = np.stack(res)
    return out if np.ndim(q) else out[0]

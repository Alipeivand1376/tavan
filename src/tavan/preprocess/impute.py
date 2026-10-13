"""Missing-value imputation for time series.

Available methods
-----------------
``linear``
    Linear interpolation in time (edges: nearest value).
``ffill`` / ``bfill``
    Last / next observation carried over.
``mean`` / ``median``
    Column statistic.
``seasonal``
    Seasonal profile (mean per phase of ``period``) plus linearly
    interpolated deviation.  Preserves daily shape inside long gaps.
``knn``
    Average of the ``k`` most similar complete periods (nan-Euclidean).
``em``
    Gaussian expectation–maximisation over period profiles: each missing
    block is replaced by its conditional expectation given the observed part.
``auto``
    ``seasonal`` when ``period`` is known, otherwise ``linear``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..registry import register
from .base import BaseTransform

METHODS = ("linear", "ffill", "bfill", "mean", "median", "seasonal", "knn", "em", "auto")


def _interp_column(col: np.ndarray) -> np.ndarray:
    mask = np.isfinite(col)
    if mask.all():
        return col
    if not mask.any():
        return np.zeros_like(col)
    idx = np.arange(col.size)
    out = col.copy()
    out[~mask] = np.interp(idx[~mask], idx[mask], col[mask])
    return out


def _ffill_column(col: np.ndarray, reverse: bool = False) -> np.ndarray:
    c = col[::-1] if reverse else col
    mask = np.isfinite(c)
    if not mask.any():
        return np.zeros_like(col)
    idx = np.where(mask, np.arange(c.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = c[idx]
    first = np.argmax(mask)
    out[:first] = c[first]  # leading gap: use first observation
    return out[::-1] if reverse else out


def _seasonal_profile(x: np.ndarray, period: int) -> np.ndarray:
    T, C = x.shape
    phase = np.arange(T) % period
    prof = np.zeros((period, C))
    for p in range(period):
        vals = x[phase == p]
        with np.errstate(all="ignore"):
            m = np.nanmean(vals, axis=0) if np.isfinite(vals).any() else np.full(C, np.nan)
        prof[p] = m
    for c in range(C):
        prof[:, c] = _interp_column(prof[:, c]) if np.isfinite(prof[:, c]).any() else 0.0
    return prof


def _profiles(x: np.ndarray, period: int):
    T, C = x.shape
    n = int(np.ceil(T / period))
    pad = n * period - T
    xp = np.concatenate([x, np.full((pad, C), np.nan)]) if pad else x
    return xp.reshape(n, period * C), pad


def _unprofile(P: np.ndarray, T: int, C: int) -> np.ndarray:
    return P.reshape(-1, C)[:T]


@register("transform", "imputer", aliases=("impute",))
class Imputer(BaseTransform):
    """Fill missing values (``NaN``) in ``(T,)`` or ``(T, C)`` series.

    Parameters
    ----------
    method:
        One of ``linear``, ``ffill``, ``bfill``, ``mean``, ``median``,
        ``seasonal``, ``knn``, ``em`` or ``auto``.
    period:
        Seasonal period in steps (e.g. 24 for hourly data with daily
        seasonality).  Required by ``seasonal``, ``knn`` and ``em``.
    k:
        Neighbours for ``knn``.
    max_iter, tol:
        EM iterations and relative tolerance.
    shrinkage:
        Covariance shrinkage towards its diagonal for ``em`` (0–1).

    Attributes
    ----------
    mask_ : ndarray of bool
        ``True`` where values were imputed during the last ``transform``.

    Examples
    --------
    >>> import numpy as np
    >>> from tavan.preprocess import Imputer
    >>> x = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    >>> Imputer("linear").fit_transform(x)
    array([1., 2., 3., 4., 5.])
    """

    _state_attrs = ("stats_", "profile_")

    def __init__(self, method: str = "auto", period: Optional[int] = None, k: int = 5,
                 max_iter: int = 50, tol: float = 1e-4, shrinkage: float = 0.1) -> None:
        self.method = method
        self.period = period
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.shrinkage = shrinkage

    def _resolved(self) -> str:
        m = self.method.lower()
        if m not in METHODS:
            raise ValueError(f"unknown imputation method {self.method!r}; choose from {METHODS}")
        if m == "auto":
            return "seasonal" if self.period else "linear"
        if m in ("seasonal", "knn", "em") and not self.period:
            raise ValueError(f"method {m!r} requires `period` (e.g. period=24 for hourly data)")
        return m

    def fit(self, x, y=None) -> "Imputer":
        arr = np.asarray(x, dtype=float)
        a2 = arr[:, None] if arr.ndim == 1 else arr.reshape(-1, arr.shape[-1])
        m = self._resolved()
        with np.errstate(all="ignore"):
            if m == "median":
                self.stats_ = np.nan_to_num(np.nanmedian(a2, axis=0))
            else:
                self.stats_ = np.nan_to_num(np.nanmean(a2, axis=0))
        self.profile_ = _seasonal_profile(a2, self.period) if self.period and arr.ndim <= 2 else None
        return self

    def transform(self, x) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        was_1d = arr.ndim == 1
        if arr.ndim == 3:  # (n, L, C): impute each window independently along time
            out = np.stack([self.transform(w) for w in arr])
            self.mask_ = ~np.isfinite(arr)
            return out
        a2 = arr[:, None] if was_1d else arr.copy()
        self.mask_ = ~np.isfinite(a2)
        if not self.mask_.any():
            return arr.copy()
        if not hasattr(self, "stats_"):
            self.fit(arr)
        m = self._resolved()
        if m == "linear":
            out = np.column_stack([_interp_column(a2[:, c]) for c in range(a2.shape[1])])
        elif m == "ffill":
            out = np.column_stack([_ffill_column(a2[:, c]) for c in range(a2.shape[1])])
        elif m == "bfill":
            out = np.column_stack([_ffill_column(a2[:, c], reverse=True) for c in range(a2.shape[1])])
        elif m in ("mean", "median"):
            out = np.where(self.mask_, self.stats_[None, :], a2)
        elif m == "seasonal":
            out = self._seasonal(a2)
        elif m == "knn":
            out = self._knn(a2)
        else:
            out = self._em(a2)
        out[~self.mask_] = a2[~self.mask_]
        return out[:, 0] if was_1d else out

    # ----------------------------------------------------------- methods
    def _seasonal(self, x: np.ndarray) -> np.ndarray:
        T, C = x.shape
        prof = self.profile_ if self.profile_ is not None and self.profile_.shape[1] == C else _seasonal_profile(x, self.period)
        base = prof[np.arange(T) % self.period]
        resid = x - base
        resid = np.column_stack([_interp_column(resid[:, c]) for c in range(C)])
        return base + resid

    def _knn(self, x: np.ndarray) -> np.ndarray:
        T, C = x.shape
        P, _ = _profiles(x, self.period)
        fallback, _ = _profiles(self._seasonal(x), self.period)
        obs = np.isfinite(P)
        complete = obs.all(axis=1)
        if complete.sum() == 0:
            return self._seasonal(x)
        donors = P[complete]
        out = P.copy()
        k = min(self.k, len(donors))
        for i in np.where(~complete)[0]:
            o = obs[i]
            if o.sum() == 0:
                out[i] = np.where(np.isfinite(out[i]), out[i], fallback[i])
                continue
            d = np.sqrt(np.mean((donors[:, o] - P[i, o]) ** 2, axis=1))
            nn = np.argsort(d)[:k]
            w = 1.0 / (d[nn] + 1e-9)
            est = (w[:, None] * donors[nn]).sum(0) / w.sum()
            # shift neighbours to the row's own level on observed entries
            est = est + np.mean(P[i, o] - est[o])
            out[i, ~o] = est[~o]
        return _unprofile(out, T, C)

    def _em(self, x: np.ndarray) -> np.ndarray:
        T, C = x.shape
        P, _ = _profiles(x, self.period)
        obs = np.isfinite(P)
        Z, _ = _profiles(self._seasonal(x), self.period)
        Z[obs] = P[obs]
        d = Z.shape[1]
        prev = None
        for _ in range(self.max_iter):
            mu = Z.mean(0)
            S = np.cov(Z, rowvar=False) if len(Z) > 1 else np.eye(d)
            S = np.atleast_2d(S)
            S = (1 - self.shrinkage) * S + self.shrinkage * np.diag(np.diag(S)) + 1e-8 * np.eye(d)
            for i in np.where(~obs.all(1))[0]:
                o, m = obs[i], ~obs[i]
                if o.sum() == 0:
                    Z[i] = mu
                    continue
                Soo = S[np.ix_(o, o)]
                Smo = S[np.ix_(m, o)]
                Z[i, m] = mu[m] + Smo @ np.linalg.solve(Soo, P[i, o] - mu[o])
            if prev is not None:
                change = np.linalg.norm(Z - prev) / (np.linalg.norm(prev) + 1e-12)
                if change < self.tol:
                    break
            prev = Z.copy()
        return _unprofile(Z, T, C)


def impute(x, method: str = "auto", period: Optional[int] = None, **kwargs) -> np.ndarray:
    """Functional shortcut for :class:`Imputer`."""
    return Imputer(method=method, period=period, **kwargs).fit_transform(x)


def missing_report(x) -> dict:
    """Summary of missingness: ratio, number of gaps, longest gap (per channel)."""
    arr = np.asarray(x, dtype=float)
    a2 = arr[:, None] if arr.ndim == 1 else arr
    rep = {"ratio": [], "n_gaps": [], "longest_gap": []}
    for c in range(a2.shape[1]):
        miss = ~np.isfinite(a2[:, c])
        rep["ratio"].append(float(miss.mean()))
        d = np.diff(np.concatenate([[0], miss.astype(int), [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        rep["n_gaps"].append(int(len(starts)))
        rep["longest_gap"].append(int((ends - starts).max()) if len(starts) else 0)
    return rep

"""Quality metrics for probabilistic scenarios and synthetic data.

Proper scoring rules (energy score, variogram score, CRPS) compare
scenarios with observations; distributional distances (MMD, Wasserstein,
autocorrelation and correlation errors, a classifier two-sample test)
compare synthetic with real samples.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def _flat(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X.reshape(X.shape[0], -1) if X.ndim > 2 else (X[:, None] if X.ndim == 1 else X)


# ---------------------------------------------------------- scoring rules
def crps_ensemble(obs, scenarios) -> float:
    """Continuous ranked probability score averaged over features.

    ``obs`` ``(d,)`` or ``(m, d)``; ``scenarios`` ``(n, d)``.  Lower is better.
    """
    S = _flat(scenarios)
    Y = np.atleast_2d(np.asarray(obs, dtype=float).reshape(-1, S.shape[1]))
    n = S.shape[0]
    Ss = np.sort(S, axis=0)
    # E|X - X'| via sorted formula: 2/n^2 * sum_i (2i - n - 1) x_(i)
    w = (2 * np.arange(1, n + 1) - n - 1)[:, None]
    e_xx = 2.0 * (w * Ss).sum(0) / n ** 2
    vals = [np.mean(np.abs(S - y[None, :]), axis=0) - 0.5 * e_xx for y in Y]
    return float(np.mean(vals))


def energy_score(obs, scenarios, max_pairs: int = 2_000_000) -> float:
    """Multivariate energy score (Gneiting & Raftery, 2007); lower is better.

    ``ES = E‖X − y‖ − ½ E‖X − X′‖`` averaged over observation rows.
    Sensitive to errors in both marginals and inter-temporal dependence.
    """
    S = _flat(scenarios)
    Y = np.atleast_2d(np.asarray(obs, dtype=float).reshape(-1, S.shape[1]))
    n = S.shape[0]
    if n * n > max_pairs:
        idx = np.random.default_rng(0).choice(n, size=int(np.sqrt(max_pairs)), replace=False)
        Sp = S[idx]
    else:
        Sp = S
    sq = (Sp ** 2).sum(1)
    dxx = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * Sp @ Sp.T, 0)).mean()
    dxy = np.sqrt(((S[None, :, :] - Y[:, None, :]) ** 2).sum(-1)).mean(1)
    return float(np.mean(dxy - 0.5 * dxx))


def variogram_score(obs, scenarios, p: float = 0.5, weights: Optional[np.ndarray] = None) -> float:
    """Variogram score of order ``p`` (Scheuerer & Hamill, 2015); lower is better.

    Focuses on the correlation structure between time steps.
    """
    S = _flat(scenarios)
    Y = np.atleast_2d(np.asarray(obs, dtype=float).reshape(-1, S.shape[1]))
    d = S.shape[1]
    W = np.ones((d, d)) if weights is None else weights
    ev = (np.abs(S[:, :, None] - S[:, None, :]) ** p).mean(0)
    vals = [np.sum(W * (np.abs(y[:, None] - y[None, :]) ** p - ev) ** 2) for y in Y]
    return float(np.mean(vals)) / (d * d)


def interval_coverage(obs, scenarios, level: float = 0.9) -> float:
    """Fraction of observed values inside the central ``level`` interval."""
    S = _flat(scenarios)
    Y = np.atleast_2d(np.asarray(obs, dtype=float).reshape(-1, S.shape[1]))
    lo, hi = np.quantile(S, [(1 - level) / 2, (1 + level) / 2], axis=0)
    return float(np.mean((Y >= lo) & (Y <= hi)))


# ------------------------------------------------- distribution distances
def mmd(X, Y, bandwidth: Optional[float] = None, max_samples: int = 1500, seed: int = 0) -> float:
    """Unbiased squared maximum mean discrepancy with an RBF kernel.

    The bandwidth defaults to the median pairwise distance.  ``≈0`` means
    the two samples are indistinguishable at the kernel's resolution.
    """
    A, B = _flat(X), _flat(Y)
    rng = np.random.default_rng(seed)
    if len(A) > max_samples:
        A = A[rng.choice(len(A), max_samples, replace=False)]
    if len(B) > max_samples:
        B = B[rng.choice(len(B), max_samples, replace=False)]
    Z = np.concatenate([A, B])
    sq = (Z ** 2).sum(1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0)
    if bandwidth is None:
        tri = D2[np.triu_indices(len(Z), 1)]
        bandwidth = float(np.sqrt(np.median(tri) / 2)) if tri.size else 1.0
        bandwidth = bandwidth if bandwidth > 0 else 1.0
    K = np.exp(-D2 / (2 * bandwidth ** 2))
    m, n = len(A), len(B)
    Kxx, Kyy, Kxy = K[:m, :m], K[m:, m:], K[:m, m:]
    t1 = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    t2 = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    return float(t1 + t2 - 2 * Kxy.mean())


def wasserstein_1d(a, b) -> float:
    """1-D Wasserstein-1 distance between two empirical distributions."""
    a = np.sort(np.asarray(a, dtype=float).ravel())
    b = np.sort(np.asarray(b, dtype=float).ravel())
    q = np.linspace(0, 1, max(a.size, b.size) + 1)[1:-1] if max(a.size, b.size) > 1 else np.array([0.5])
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def marginal_wasserstein(X, Y) -> float:
    """Average per-feature Wasserstein-1 distance."""
    A, B = _flat(X), _flat(Y)
    return float(np.mean([wasserstein_1d(A[:, j], B[:, j]) for j in range(A.shape[1])]))


def acf(X, nlags: int = 10) -> np.ndarray:
    """Average autocorrelation function of the rows of ``X`` (each row a series)."""
    A = _flat(X)
    A = A - A.mean(1, keepdims=True)
    den = (A * A).sum(1) + 1e-12
    nlags = min(nlags, A.shape[1] - 1)
    return np.array([np.mean((A[:, k:] * A[:, : A.shape[1] - k]).sum(1) / den) for k in range(nlags + 1)])


def acf_error(X, Y, nlags: int = 10) -> float:
    """Mean absolute difference of average ACFs (temporal dynamics)."""
    return float(np.mean(np.abs(acf(X, nlags) - acf(Y, nlags))))


def correlation_error(X, Y) -> float:
    """Mean absolute difference of the feature correlation matrices."""
    A, B = _flat(X), _flat(Y)
    ca = np.nan_to_num(np.corrcoef(A, rowvar=False))
    cb = np.nan_to_num(np.corrcoef(B, rowvar=False))
    return float(np.mean(np.abs(ca - cb)))


def discriminative_score(X, Y, test_fraction: float = 0.3, epochs: int = 300, seed: int = 0) -> float:
    """Classifier two-sample test: ``|accuracy - 0.5|`` of a logistic classifier.

    Uses standardised features plus their pairwise differences (ramps).
    ``0`` = synthetic data indistinguishable from real; ``0.5`` = trivially
    separable.
    """
    A, B = _flat(X), _flat(Y)
    rng = np.random.default_rng(seed)
    m = min(len(A), len(B))
    A, B = A[rng.choice(len(A), m, replace=False)], B[rng.choice(len(B), m, replace=False)]

    def feats(Z):
        return np.concatenate([Z, np.diff(Z, axis=1), np.sort(Z, axis=1)], axis=1) if Z.shape[1] > 1 else Z

    F = np.concatenate([feats(A), feats(B)])
    y = np.concatenate([np.ones(m), np.zeros(m)])
    mu, sd = F.mean(0), F.std(0) + 1e-9
    F = (F - mu) / sd
    perm = rng.permutation(2 * m)
    F, y = F[perm], y[perm]
    k = max(2, int(2 * m * test_fraction))
    Ft, yt, Fv, yv = F[k:], y[k:], F[:k], y[:k]
    w = np.zeros(F.shape[1])
    b = 0.0
    lr, lam = 0.1, 1e-2
    for _ in range(epochs):
        z = Ft @ w + b
        pr = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        g = pr - yt
        w -= lr * (Ft.T @ g / len(yt) + lam * w)
        b -= lr * g.mean()
    acc = np.mean(((Fv @ w + b) > 0) == (yv > 0.5))
    return float(abs(acc - 0.5))


DEFAULT_METRICS = ("energy_score", "mmd", "wasserstein", "acf_error", "correlation_error", "discriminative")


def evaluate_scenarios(real, generated, metrics: Sequence[str] = DEFAULT_METRICS, nlags: int = 10) -> Dict[str, float]:
    """Compare generated scenarios with real samples.

    Parameters
    ----------
    real:
        Held-out real samples ``(m, H[, C])``.
    generated:
        Synthetic samples ``(n, H[, C])``.
    metrics:
        Any of ``energy_score``, ``variogram_score``, ``crps``, ``mmd``,
        ``wasserstein``, ``acf_error``, ``correlation_error``,
        ``discriminative``, ``coverage``.
    """
    R, G = _flat(real), _flat(generated)
    funcs = {
        "energy_score": lambda: energy_score(R, G),
        "variogram_score": lambda: variogram_score(R, G),
        "crps": lambda: crps_ensemble(R, G),
        "mmd": lambda: mmd(R, G),
        "wasserstein": lambda: marginal_wasserstein(R, G),
        "acf_error": lambda: acf_error(R, G, nlags),
        "correlation_error": lambda: correlation_error(R, G),
        "discriminative": lambda: discriminative_score(R, G),
        "coverage": lambda: interval_coverage(R, G, 0.9),
    }
    out = {}
    for m in metrics:
        if m not in funcs:
            raise ValueError(f"unknown metric {m!r}; choose from {sorted(funcs)}")
        out[m] = funcs[m]()
    return out

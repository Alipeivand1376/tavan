"""Risk measures on discrete scenario sets.

All functions accept per-scenario *losses* (larger = worse) and optional
probabilities.  ``alpha`` is the confidence level, e.g. ``0.95`` means the
worst 5 % of probability mass forms the tail.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import numpy as np

LossSpec = Union[str, Callable[[np.ndarray], np.ndarray], None]


def _probs(n: int, p) -> np.ndarray:
    if p is None:
        return np.full(n, 1.0 / n)
    p = np.asarray(p, dtype=float)
    if p.shape != (n,) or np.any(p < 0):
        raise ValueError("probabilities must be non-negative with one entry per scenario")
    return p / p.sum()


def value_at_risk(losses, alpha: float = 0.95, probs=None) -> float:
    """VaR: smallest ``l`` with ``P(L ≤ l) ≥ alpha``."""
    L = np.asarray(losses, dtype=float).ravel()
    p = _probs(L.size, probs)
    order = np.argsort(L)
    cp = np.cumsum(p[order])
    i = int(np.searchsorted(cp, alpha - 1e-12))
    return float(L[order][min(i, L.size - 1)])


def conditional_value_at_risk(losses, alpha: float = 0.95, probs=None) -> float:
    """CVaR (expected shortfall) by the Rockafellar–Uryasev formula.

    ``CVaR_α = VaR_α + E[(L - VaR_α)^+] / (1 - α)``, exact for discrete
    distributions including fractional atoms at the VaR.
    """
    L = np.asarray(losses, dtype=float).ravel()
    p = _probs(L.size, probs)
    var = value_at_risk(L, alpha, p)
    if alpha >= 1:
        return float(L.max())
    return float(var + np.sum(p * np.maximum(L - var, 0.0)) / (1 - alpha))


var = value_at_risk
cvar = conditional_value_at_risk


def scenario_loss(X: np.ndarray, loss: LossSpec = "sum", channel: Optional[int] = None) -> np.ndarray:
    """Reduce scenarios ``(n, H[, C])`` to one scalar loss each.

    ``loss`` may be ``"sum"`` (energy), ``"mean"``, ``"max"`` (peak),
    ``"min"``, ``"neg_sum"`` (shortfall of generation), ``"ramp"`` (largest
    absolute ramp), ``"std"`` or a callable ``f(X) -> (n,)``.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 3:
        X = X[:, :, channel] if channel is not None else X.sum(axis=2)
    if callable(loss):
        out = np.asarray(loss(X), dtype=float)
        if out.shape != (X.shape[0],):
            raise ValueError("custom loss must return one value per scenario")
        return out
    table = {
        "sum": lambda a: a.sum(1),
        "mean": lambda a: a.mean(1),
        "max": lambda a: a.max(1),
        "peak": lambda a: a.max(1),
        "min": lambda a: a.min(1),
        "neg_sum": lambda a: -a.sum(1),
        "neg_min": lambda a: -a.min(1),
        "ramp": lambda a: np.abs(np.diff(a, axis=1)).max(1),
        "std": lambda a: a.std(1),
    }
    if loss is None:
        loss = "sum"
    if loss not in table:
        raise ValueError(f"unknown loss {loss!r}; choose from {sorted(table)} or pass a callable")
    return table[loss](X)


def stratified_tail_sample(losses: np.ndarray, n: int, alpha: float = 0.95, tail_share: float = 0.5,
                           rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Select ``n`` of a large pool so that the tail is over-represented.

    The pool is split into body (``P = α``) and tail (``P = 1 - α``) by the
    VaR.  ``tail_share`` of the ``n`` slots go to the tail; probabilities are
    re-weighted so that each stratum keeps its original mass.  Expectations
    stay unbiased while CVaR estimates use many more tail scenarios than
    plain Monte-Carlo with the same budget.

    Returns
    -------
    idx : selected indices into the pool
    probs : probabilities of the selected scenarios (sum to 1)
    """
    rng = rng or np.random.default_rng()
    L = np.asarray(losses, dtype=float)
    N = L.size
    order = np.argsort(L)
    n_tail_pool = max(1, int(round(N * (1 - alpha))))
    tail_pool, body_pool = order[N - n_tail_pool:], order[: N - n_tail_pool]
    n_tail = int(np.clip(round(n * tail_share), 1, min(n - 1, tail_pool.size)))
    n_body = n - n_tail
    if body_pool.size < n_body:
        n_body = body_pool.size
        n_tail = min(n - n_body, tail_pool.size)
    ti = rng.choice(tail_pool, size=n_tail, replace=False)
    bi = rng.choice(body_pool, size=n_body, replace=False) if n_body else np.array([], dtype=int)
    m_tail = tail_pool.size / N
    probs = np.concatenate([np.full(n_body, (1 - m_tail) / max(n_body, 1)), np.full(n_tail, m_tail / n_tail)])
    return np.concatenate([bi, ti]).astype(int), probs / probs.sum()


def risk_summary(losses, probs=None, alphas=(0.9, 0.95, 0.99)) -> dict:
    """Mean, standard deviation, VaR and CVaR at several levels."""
    L = np.asarray(losses, dtype=float)
    p = _probs(L.size, probs)
    mu = float(np.sum(p * L))
    out = {"mean": mu, "std": float(np.sqrt(np.sum(p * (L - mu) ** 2)))}
    for a in alphas:
        out[f"VaR_{a:g}"] = value_at_risk(L, a, p)
        out[f"CVaR_{a:g}"] = conditional_value_at_risk(L, a, p)
    return out

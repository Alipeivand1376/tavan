"""Scenario reduction: k-medoids, fast forward selection and a risk-preserving variant.

Stochastic optimisation models (unit commitment, storage scheduling,
stochastic OPF) become intractable with thousands of scenarios.  These
methods select a small representative subset and redistribute
probabilities, minimising the (approximate) Kantorovich distance to the
original distribution.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .risk import _probs, value_at_risk


def _flat(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X.reshape(X.shape[0], -1)


def pairwise_distances(A: np.ndarray, B: Optional[np.ndarray] = None, p: int = 2) -> np.ndarray:
    """Euclidean (``p=2``) or Manhattan (``p=1``) distance matrix."""
    A = _flat(A)
    B = A if B is None else _flat(B)
    if p == 1:
        return np.abs(A[:, None, :] - B[None, :, :]).sum(-1)
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.sqrt(np.maximum(sq, 0.0))


def _assign(D: np.ndarray, sel: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lab = np.argmin(D[:, sel], axis=1)
    newp = np.bincount(lab, weights=p, minlength=len(sel))
    return lab, newp


def kantorovich_distance(X, probs, idx, new_probs=None) -> float:
    """Transport cost from the full set to the subset ``idx`` (nearest-point assignment)."""
    D = pairwise_distances(X, np.asarray(X)[idx])
    p = _probs(len(D), probs)
    return float(np.sum(p * D.min(axis=1)))


def kmedoids(X, k: int, probs=None, max_iter: int = 100, seed: Optional[int] = 0,
             D: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Probability-weighted k-medoids (alternating / Voronoi iteration).

    Returns medoid indices and the aggregated probabilities of their clusters.
    """
    n = len(X)
    k = int(min(k, n))
    p = _probs(n, probs)
    D = pairwise_distances(X) if D is None else D
    rng = np.random.default_rng(seed)
    # weighted k-means++ initialisation
    sel = [int(rng.choice(n, p=p))]
    d2 = D[sel[0]] ** 2
    for _ in range(1, k):
        w = p * d2
        sel.append(int(rng.choice(n, p=w / w.sum())) if w.sum() > 0 else int(rng.integers(n)))
        d2 = np.minimum(d2, D[sel[-1]] ** 2)
    sel = np.array(sel)
    for _ in range(max_iter):
        lab = np.argmin(D[:, sel], axis=1)
        new = sel.copy()
        for j in range(k):
            members = np.where(lab == j)[0]
            if members.size == 0:
                continue
            cost = (D[np.ix_(members, members)] * p[members][:, None]).sum(0)
            new[j] = members[np.argmin(cost)]
        if np.array_equal(np.sort(new), np.sort(sel)):
            sel = new
            break
        sel = new
    _, newp = _assign(D, sel, p)
    return sel, newp


def fast_forward_selection(X, k: int, probs=None, D: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Fast forward selection of Heitsch & Römisch (2003).

    Greedily adds the scenario that most reduces the Kantorovich distance;
    deterministic and usually better than random or k-means selection.
    """
    n = len(X)
    k = int(min(k, n))
    p = _probs(n, probs)
    D = pairwise_distances(X) if D is None else D.copy()
    selected = []
    remaining = np.ones(n, dtype=bool)
    best_dist = np.full(n, np.inf)
    for _ in range(k):
        cand = np.where(remaining)[0]
        # cost of adding u: sum_i p_i * min(best_dist_i, D_iu), excluding u itself
        m = np.minimum(best_dist[:, None], D[:, cand])
        cost = (p[:, None] * m).sum(0) - p[cand] * m[cand, np.arange(cand.size)]
        u = cand[int(np.argmin(cost))]
        selected.append(u)
        remaining[u] = False
        best_dist = np.minimum(best_dist, D[:, u])
    sel = np.array(selected)
    _, newp = _assign(D, sel, p)
    return sel, newp


def risk_preserving_reduction(X, losses, k: int, alpha: float = 0.95, probs=None, tail_fraction: float = 0.3,
                              method: str = "kmedoids", seed: Optional[int] = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce scenarios while preserving the loss tail (CVaR).

    Body and tail (split at ``VaR_alpha`` of ``losses``) are reduced
    separately, with ``tail_fraction`` of the ``k`` representatives reserved
    for the tail, so that the CVaR of the reduced set stays close to that of
    the original — plain reduction tends to average extreme scenarios away.
    """
    n = len(X)
    p = _probs(n, probs)
    L = np.asarray(losses, dtype=float)
    v = value_at_risk(L, alpha, p)
    tail = np.where(L > v)[0]
    if tail.size == 0:
        tail = np.where(L >= v)[0]
    body = np.setdiff1d(np.arange(n), tail)
    k_tail = int(np.clip(round(k * tail_fraction), 1, max(1, min(k - 1, tail.size))))
    k_body = k - k_tail
    reducer = kmedoids if method == "kmedoids" else fast_forward_selection
    parts_idx, parts_p = [], []
    for group, kk in ((body, k_body), (tail, k_tail)):
        if group.size == 0 or kk <= 0:
            continue
        mass = p[group].sum()
        if method == "kmedoids":
            s, sp = reducer(np.asarray(X)[group], kk, p[group] / mass, seed=seed)
        else:
            s, sp = reducer(np.asarray(X)[group], kk, p[group] / mass)
        parts_idx.append(group[s])
        parts_p.append(sp * mass)
    idx = np.concatenate(parts_idx)
    newp = np.concatenate(parts_p)
    return idx, newp / newp.sum()


def reduce_scenarios(X, k: int, method: str = "kmedoids", probs=None, losses=None, alpha: float = 0.95,
                     seed: Optional[int] = 0, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch to a reduction method.

    ``method`` is ``"kmedoids"``, ``"forward"`` (fast forward selection),
    ``"cvar"`` (risk-preserving; needs ``losses``) or ``"random"``.
    """
    if method in ("kmedoids", "kmedoid", "medoids"):
        return kmedoids(X, k, probs, seed=seed, **kwargs)
    if method in ("forward", "ffs", "fast_forward"):
        return fast_forward_selection(X, k, probs)
    if method in ("cvar", "risk", "tail"):
        if losses is None:
            raise ValueError("method='cvar' requires losses")
        return risk_preserving_reduction(X, losses, k, alpha, probs, seed=seed, **kwargs)
    if method == "random":
        n = len(X)
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=min(k, n), replace=False, p=_probs(n, probs))
        return idx, np.full(len(idx), 1.0 / len(idx))
    raise ValueError(f"unknown reduction method {method!r}")

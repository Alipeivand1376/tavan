"""Distribution-drift detection for deciding when to re-train generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .._stats import ks_2samp
from .metrics import _flat, mmd


@dataclass
class DriftResult:
    """Outcome of a drift test."""

    statistic: float
    p_value: float
    drift: bool
    method: str

    def __bool__(self) -> bool:
        return self.drift


class DriftDetector:
    """Two-sample test between a reference window and new data.

    Parameters
    ----------
    reference:
        Reference samples ``(n, d)`` (e.g. the training profiles).
    method:
        ``"mmd"`` (kernel two-sample test with permutation p-value),
        ``"energy"`` (energy-distance permutation test) or ``"ks"``
        (per-feature Kolmogorov–Smirnov with Bonferroni correction).
    alpha:
        Significance level.
    n_permutations:
        Permutations for the MMD / energy tests.
    """

    def __init__(self, reference, method: str = "mmd", alpha: float = 0.05, n_permutations: int = 200,
                 seed: Optional[int] = 0) -> None:
        self.reference = _flat(reference)
        self.method = method
        self.alpha = alpha
        self.n_permutations = n_permutations
        self.seed = seed

    def _stat(self, A, B):
        if self.method == "mmd":
            return mmd(A, B, bandwidth=self._bw)
        return _energy_distance(A, B)

    def test(self, new) -> DriftResult:
        B = _flat(new)
        A = self.reference
        if self.method == "ks":
            pvals = [ks_2samp(A[:, j], B[:, j])[1] for j in range(A.shape[1])]
            pmin = min(1.0, min(pvals) * A.shape[1])
            stat = max(ks_2samp(A[:, j], B[:, j])[0] for j in range(A.shape[1]))
            return DriftResult(float(stat), float(pmin), pmin < self.alpha, "ks")
        rng = np.random.default_rng(self.seed)
        if len(A) > 500:
            A = A[rng.choice(len(A), 500, replace=False)]
        Z = np.concatenate([A, B])
        sq = (Z ** 2).sum(1)
        D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0)
        self._bw = float(np.sqrt(np.median(D2[np.triu_indices(len(Z), 1)]) / 2)) or 1.0
        obs = self._stat(A, B)
        m = len(A)
        count = 0
        for _ in range(self.n_permutations):
            p = rng.permutation(len(Z))
            if self._stat(Z[p[:m]], Z[p[m:]]) >= obs:
                count += 1
        pval = (count + 1) / (self.n_permutations + 1)
        return DriftResult(float(obs), float(pval), pval < self.alpha, self.method)


def _energy_distance(A, B) -> float:
    def mean_dist(X, Y):
        sx, sy = (X ** 2).sum(1), (Y ** 2).sum(1)
        return np.sqrt(np.maximum(sx[:, None] + sy[None, :] - 2 * X @ Y.T, 0)).mean()

    return float(2 * mean_dist(A, B) - mean_dist(A, A) - mean_dist(B, B))


class PageHinkley:
    """Streaming Page–Hinkley test for an increase in the mean of a signal.

    Feed forecast errors or anomaly scores one at a time with
    :meth:`update`; it returns ``True`` when a change is detected.
    """

    def __init__(self, delta: float = 0.005, threshold: float = 50.0, burn_in: int = 30) -> None:
        self.delta = delta
        self.threshold = threshold
        self.burn_in = burn_in
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.cum = 0.0
        self.min_cum = 0.0

    def update(self, value: float) -> bool:
        self.n += 1
        self.mean += (value - self.mean) / self.n
        self.cum += value - self.mean - self.delta
        self.min_cum = min(self.min_cum, self.cum)
        return self.n > self.burn_in and (self.cum - self.min_cum) > self.threshold

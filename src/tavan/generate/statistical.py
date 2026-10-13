"""Fast statistical generators: analog bootstrap, Gaussian copula, GMM.

They train in milliseconds, are fully transparent and serve as strong
baselines for the neural generators.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .._stats import norm_cdf, norm_ppf
from ..registry import register
from .base import BaseGenerator


# ===================================================================== bootstrap
@register("generator", "bootstrap", aliases=("analog", "resample"))
class Bootstrap(BaseGenerator):
    """Analog / bootstrap resampling of historical samples.

    Without conditions, samples are drawn uniformly with replacement.  With
    conditions, each draw picks one of the ``k`` historical samples whose
    conditions are closest (an *analog ensemble*).

    Parameters
    ----------
    k:
        Number of analogs considered per condition.
    jitter:
        Standard deviation of Gaussian noise added, as a fraction of the
        per-feature standard deviation (0 keeps exact historical samples).
    seed:
        Default random seed.
    """

    _state_attrs = ("X_", "cond_", "std_")

    def __init__(self, k: int = 10, jitter: float = 0.0, seed: Optional[int] = None) -> None:
        self.k = k
        self.jitter = jitter
        self.seed = seed

    def _fit(self, X, cond):
        self.X_ = X.copy()
        self.cond_ = None if cond is None else cond.copy()
        self.std_ = X.std(axis=0)

    def _get_state(self):
        st = super()._get_state()
        st.update(n_features_=self.n_features_, n_cond_=self.n_cond_, is_fitted_=True)
        return st

    def _sample(self, n, cond, rng):
        if cond is None:
            idx = rng.integers(0, len(self.X_), size=n)
        else:
            d = ((cond[:, None, :] - self.cond_[None, :, :]) ** 2).sum(-1)
            k = min(self.k, len(self.X_))
            nn = np.argpartition(d, k - 1, axis=1)[:, :k]
            idx = nn[np.arange(n), rng.integers(0, k, size=n)]
        out = self.X_[idx].copy()
        if self.jitter:
            out += rng.normal(size=out.shape) * self.jitter * self.std_
        return out


# ==================================================================== copula
@register("generator", "copula", aliases=("gaussian_copula",))
class GaussianCopula(BaseGenerator):
    """Gaussian copula with empirical marginals.

    Each feature is mapped to normal scores through its empirical CDF; the
    dependence between features (e.g. hours of the day) is captured by the
    correlation matrix of the scores.  Conditions are modelled jointly and
    handled by exact Gaussian conditioning in score space.

    Parameters
    ----------
    shrinkage:
        Ledoit–Wolf-style shrinkage of the correlation matrix towards the
        identity (stabilises small samples).
    n_quantiles:
        Resolution of the stored marginal quantile functions.
    tail_extrapolation:
        Fraction of the observed range by which samples may exceed the
        historical minimum/maximum (0 = never).
    """

    supports_likelihood = False
    _state_attrs = ("q_grid_", "q_values_", "sorted_", "corr_", "n_features_", "n_cond_", "is_fitted_")

    def __init__(self, shrinkage: float = 0.05, n_quantiles: int = 512, tail_extrapolation: float = 0.05,
                 seed: Optional[int] = None) -> None:
        self.shrinkage = shrinkage
        self.n_quantiles = n_quantiles
        self.tail_extrapolation = tail_extrapolation
        self.seed = seed

    def _fit(self, X, cond):
        Z = X if cond is None else np.concatenate([X, cond], axis=1)
        n, d = Z.shape
        nq = min(self.n_quantiles, n)
        self.q_grid_ = np.linspace(0.5 / n, 1 - 0.5 / n, nq)
        self.q_values_ = np.quantile(Z, self.q_grid_, axis=0)  # (nq, d)
        self.sorted_ = np.sort(Z, axis=0)
        scores = norm_ppf(self._midrank(Z, slice(None)))
        C = np.corrcoef(scores, rowvar=False) if d > 1 else np.ones((1, 1))
        C = np.atleast_2d(np.nan_to_num(C))
        np.fill_diagonal(C, 1.0)
        self.corr_ = (1 - self.shrinkage) * C + self.shrinkage * np.eye(d)

    def _midrank(self, values: np.ndarray, cols) -> np.ndarray:
        """Empirical CDF with mid-ranks for ties (robust for binary conditions)."""
        S = self.sorted_[:, cols]
        n = S.shape[0]
        u = np.column_stack([
            (np.searchsorted(S[:, j], values[:, j], "left") + np.searchsorted(S[:, j], values[:, j], "right")) / (2.0 * n)
            for j in range(values.shape[1])])
        return np.clip(u, 0.5 / n, 1 - 0.5 / n)

    def _to_scores(self, values: np.ndarray, cols: slice) -> np.ndarray:
        return norm_ppf(self._midrank(values, cols))

    def _from_scores(self, z: np.ndarray) -> np.ndarray:
        u = norm_cdf(z)
        d = z.shape[1]
        qv = self.q_values_[:, :d]
        lo, hi = qv[0], qv[-1]
        span = (hi - lo) * self.tail_extrapolation
        g = self.q_grid_
        out = np.empty_like(u)
        for j in range(d):
            x = np.interp(u[:, j], g, qv[:, j])
            # linear tail extrapolation beyond observed quantiles
            below, above = u[:, j] < g[0], u[:, j] > g[-1]
            if span[j] > 0:
                x[below] = lo[j] - span[j] * (g[0] - u[below, j]) / g[0]
                x[above] = hi[j] + span[j] * (u[above, j] - g[-1]) / (1 - g[-1])
            out[:, j] = x
        return out

    def _sample(self, n, cond, rng):
        d = self.n_features_
        C = self.corr_
        if cond is None:
            L = np.linalg.cholesky(C[:d, :d] + 1e-10 * np.eye(d))
            z = rng.normal(size=(n, d)) @ L.T
        else:
            zc = self._to_scores(cond, slice(d, None))
            Sxx, Sxc, Scc = C[:d, :d], C[:d, d:], C[d:, d:]
            K = np.linalg.solve(Scc + 1e-8 * np.eye(Scc.shape[0]), Sxc.T).T  # (d, k)
            mu = zc @ K.T
            S = Sxx - K @ Sxc.T
            L = np.linalg.cholesky((S + S.T) / 2 + 1e-8 * np.eye(d))
            z = mu + rng.normal(size=(n, d)) @ L.T
        return self._from_scores(z)


# ======================================================================= GMM
def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def _gauss_logpdf(X: np.ndarray, mu: np.ndarray, chol: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    diff = X - mu
    sol = np.linalg.solve(chol, diff.T).T
    maha = (sol ** 2).sum(1)
    logdet = 2 * np.log(np.diag(chol)).sum()
    return -0.5 * (d * np.log(2 * np.pi) + logdet + maha)


@register("generator", "gmm", aliases=("gaussian_mixture",))
class GaussianMixture(BaseGenerator):
    """Gaussian mixture model fitted by expectation–maximisation.

    Daily load or PV profiles often form a few typical *shapes* (working
    days, weekends, cloudy days); a mixture captures this multi-modality.
    Conditions are fitted jointly and handled by analytic conditioning,
    which updates both the mixture weights and the component means.

    Parameters
    ----------
    n_components:
        Number of components or ``"auto"`` (choose by BIC up to
        ``max_components``).
    covariance:
        ``"full"`` or ``"diag"``.
    reg_covar:
        Ridge added to covariance diagonals.
    n_init:
        Number of k-means++ initialisations; the best likelihood is kept.
    """

    supports_likelihood = True
    _state_attrs = ("weights_", "means_", "covs_", "n_features_", "n_cond_", "is_fitted_",
                    "bic_", "n_components_")

    def __init__(self, n_components: Union[int, str] = "auto", covariance: str = "full", reg_covar: float = 1e-4,
                 max_iter: int = 300, tol: float = 1e-5, n_init: int = 2, max_components: int = 10,
                 seed: Optional[int] = None) -> None:
        self.n_components = n_components
        self.covariance = covariance
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.max_components = max_components
        self.seed = seed

    # ---------------------------------------------------------------- EM
    def _kmeanspp(self, X, k, rng):
        n = len(X)
        centers = [X[rng.integers(n)]]
        d2 = ((X - centers[0]) ** 2).sum(1)
        for _ in range(1, k):
            p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
            centers.append(X[rng.choice(n, p=p)])
            d2 = np.minimum(d2, ((X - centers[-1]) ** 2).sum(1))
        return np.array(centers)

    def _em(self, X, k, rng):
        n, d = X.shape
        mu = self._kmeanspp(X, k, rng)
        lab = np.argmin(((X[:, None] - mu[None]) ** 2).sum(-1), axis=1)
        R = np.eye(k)[lab] * 0.9 + 0.1 / k
        ll_old = -np.inf
        reg = self.reg_covar * np.eye(d)
        for _ in range(self.max_iter):
            Nk = R.sum(0) + 1e-10
            w = Nk / n
            mu = (R.T @ X) / Nk[:, None]
            covs = np.empty((k, d, d))
            for j in range(k):
                diff = X - mu[j]
                if self.covariance == "diag":
                    covs[j] = np.diag((R[:, j] @ diff ** 2) / Nk[j]) + reg
                else:
                    covs[j] = (R[:, j, None] * diff).T @ diff / Nk[j] + reg
            chols = np.array([np.linalg.cholesky(c) for c in covs])
            logp = np.column_stack([_gauss_logpdf(X, mu[j], chols[j]) for j in range(k)]) + np.log(w)
            ll_i = _logsumexp(logp, 1)
            ll = ll_i.mean()
            R = np.exp(logp - ll_i[:, None])
            if abs(ll - ll_old) < self.tol * max(1.0, abs(ll)):
                break
            ll_old = ll
        return w, mu, covs, ll * n

    def _n_params(self, k, d):
        cov_p = d * (d + 1) / 2 if self.covariance == "full" else d
        return k * (d + cov_p) + k - 1

    def _fit(self, X, cond):
        Z = X if cond is None else np.concatenate([X, cond], axis=1)
        n, d = Z.shape
        rng = np.random.default_rng(self.seed)
        if self.n_components == "auto":
            ks = range(1, max(1, min(self.max_components, n // max(2, d // 4 + 1))) + 1)
        else:
            ks = [int(self.n_components)]
        best = None
        self.bic_ = {}
        for k in ks:
            if k > n:
                break
            cand = max((self._em(Z, k, rng) for _ in range(self.n_init)), key=lambda r: r[3])
            bic = -2 * cand[3] + self._n_params(k, d) * np.log(n)
            self.bic_[k] = float(bic)
            if best is None or bic < best[0]:
                best = (bic, cand)
        _, (w, mu, covs, _) = best
        self.weights_, self.means_, self.covs_ = w, mu, covs
        self.n_components_ = len(w)

    # ------------------------------------------------------------ sample
    def _conditional(self, cond):
        d = self.n_features_
        k = len(self.weights_)
        n = cond.shape[0]
        logw = np.empty((n, k))
        means = np.empty((n, k, d))
        chols = []
        for j in range(k):
            m, S = self.means_[j], self.covs_[j]
            Sxx, Sxc, Scc = S[:d, :d], S[:d, d:], S[d:, d:]
            Lc = np.linalg.cholesky(Scc)
            logw[:, j] = np.log(self.weights_[j]) + _gauss_logpdf(cond, m[d:], Lc)
            K = np.linalg.solve(Scc, Sxc.T).T
            means[:, j] = m[:d] + (cond - m[d:]) @ K.T
            Sc = Sxx - K @ Sxc.T
            chols.append(np.linalg.cholesky((Sc + Sc.T) / 2 + 1e-9 * np.eye(d)))
        logw -= _logsumexp(logw, 1)[:, None]
        return np.exp(logw), means, chols

    def _sample(self, n, cond, rng):
        d = self.n_features_
        if cond is None:
            comp = rng.choice(len(self.weights_), size=n, p=self.weights_ / self.weights_.sum())
            out = np.empty((n, d))
            for j in np.unique(comp):
                idx = comp == j
                L = np.linalg.cholesky(self.covs_[j][:d, :d])
                out[idx] = self.means_[j][:d] + rng.normal(size=(idx.sum(), d)) @ L.T
            return out
        w, means, chols = self._conditional(cond)
        u = rng.random(n)[:, None]
        comp = (np.cumsum(w, 1) < u).sum(1).clip(0, len(chols) - 1)
        eps = rng.normal(size=(n, d))
        out = np.empty((n, d))
        for j in np.unique(comp):
            idx = comp == j
            out[idx] = means[idx, j] + eps[idx] @ chols[j].T
        return out

    def score_samples(self, X, cond=None) -> np.ndarray:
        """Log-density of each sample (marginal over conditions if omitted)."""
        X = np.asarray(X, dtype=float)
        d = self.n_features_
        if cond is None:
            parts = [np.log(self.weights_[j]) + _gauss_logpdf(X, self.means_[j][:d],
                                                              np.linalg.cholesky(self.covs_[j][:d, :d]))
                     for j in range(len(self.weights_))]
            return _logsumexp(np.column_stack(parts), 1)
        cond = np.atleast_2d(np.asarray(cond, dtype=float))
        w, means, chols = self._conditional(cond)
        parts = [np.log(w[:, j] + 1e-300) + np.array([
            _gauss_logpdf(X[i:i + 1], means[i, j], chols[j])[0] for i in range(len(X))])
            for j in range(len(chols))]
        return _logsumexp(np.column_stack(parts), 1)

    def predict_component(self, X) -> np.ndarray:
        """Most likely mixture component (cluster label) of each sample."""
        X = np.asarray(X, dtype=float)
        d = self.n_features_
        logp = np.column_stack([np.log(self.weights_[j]) + _gauss_logpdf(
            X, self.means_[j][:d], np.linalg.cholesky(self.covs_[j][:d, :d])) for j in range(len(self.weights_))])
        return np.argmax(logp, axis=1)

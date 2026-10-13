"""DC weighted-least-squares state estimation and bad-data detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .._stats import chi2_ppf, chi2_sf


@dataclass
class EstimationResult:
    """Output of a state-estimation solve."""

    state: np.ndarray            # estimated non-slack angles
    estimate: np.ndarray         # fitted measurements H @ state
    residual: np.ndarray         # z - estimate
    objective: float             # weighted sum of squared residuals
    dof: int                     # degrees of freedom

    def normalized_residuals(self, H: np.ndarray, R_inv: np.ndarray) -> np.ndarray:
        """Normalised residuals used to pinpoint the worst measurement."""
        W = R_inv
        try:
            G = H.T @ W @ H
            S = np.linalg.inv(W) - H @ np.linalg.solve(G, H.T)
            denom = np.sqrt(np.clip(np.diag(S), 1e-12, None))
        except np.linalg.LinAlgError:  # pragma: no cover
            denom = np.ones_like(self.residual)
        return self.residual / denom


class DCStateEstimator:
    """DC weighted-least-squares estimator with a chi-square bad-data test.

    Parameters
    ----------
    network:
        A :class:`~tavan.power.PowerNetwork`.
    measurements:
        Which quantities are metered (passed to ``measurement_matrix``).
    noise_std:
        Standard deviation of measurement noise (MW); scalar or per meter.
    alpha:
        Significance level of the chi-square detector (false-alarm rate).
    """

    def __init__(self, network, measurements=("flow", "injection"), noise_std: float = 1.0,
                 alpha: float = 0.01) -> None:
        self.network = network
        self.H, self.labels = network.measurement_matrix(measurements)
        self.m, self.n = self.H.shape
        self.noise_std = noise_std
        self.alpha = alpha
        sig = np.full(self.m, noise_std) if np.isscalar(noise_std) else np.asarray(noise_std, float)
        self.R_inv = np.diag(1.0 / sig ** 2)
        G = self.H.T @ self.R_inv @ self.H
        self.gain_inv = np.linalg.inv(G)
        self.K = self.gain_inv @ self.H.T @ self.R_inv   # state = K @ z
        self.dof = max(self.m - self.n, 1)
        self.threshold = chi2_ppf(1 - alpha, self.dof)

    def estimate(self, z: np.ndarray) -> EstimationResult:
        """Estimate state from measurement vector(s) ``z``."""
        z = np.asarray(z, dtype=float)
        state = z @ self.K.T
        fit = state @ self.H.T
        r = z - fit
        J = np.einsum("...i,ij,...j->...", r, self.R_inv, r)
        return EstimationResult(state, fit, r, float(np.atleast_1d(J)[0]) if z.ndim == 1 else J,
                                self.dof)

    def detect(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Chi-square bad-data detector.

        Returns ``(alarm, p_value)`` where ``alarm`` is ``True`` when the
        weighted residual exceeds the chi-square threshold.
        """
        res = self.estimate(z)
        J = np.atleast_1d(res.objective if np.ndim(z) == 1 else
                          np.einsum("...i,ij,...j->...", res.residual, self.R_inv, res.residual))
        alarm = J > self.threshold
        p = chi2_sf(J, self.dof)
        if np.ndim(z) == 1:
            return bool(alarm[0]), float(p[0])
        return alarm, p

    def largest_normalized_residual(self, z: np.ndarray) -> Tuple[int, float]:
        """Return the index and value of the largest normalised residual (LNR test)."""
        res = self.estimate(np.atleast_2d(z)[0])
        rn = np.abs(res.normalized_residuals(self.H, self.R_inv))
        i = int(np.argmax(rn))
        return i, float(rn[i])

    def simulate_measurements(self, p_inject: np.ndarray, seed: Optional[int] = None,
                              measurements=("flow", "injection")) -> np.ndarray:
        """Generate noisy measurements consistent with the DC model."""
        rng = np.random.default_rng(seed)
        pf = self.network.dc_power_flow(p_inject)
        parts = []
        if "flow" in measurements:
            parts.append(np.atleast_2d(pf["flow"]))
        if "injection" in measurements:
            ns = self.network._noslack
            theta = np.atleast_2d(pf["theta"])
            inj = theta @ (self.network.Bbus[np.ix_(ns, np.arange(self.network.n_bus))].T) * self.network.base_mva
            parts.append(inj)
        z = np.concatenate(parts, axis=1)
        sig = np.full(self.m, self.noise_std) if np.isscalar(self.noise_std) else self.noise_std
        z = z + rng.normal(0, 1, z.shape) * sig
        return z[0] if np.ndim(p_inject) == 1 else z

"""False-data-injection (FDI) attack generation for DC state estimation.

Liu, Ning & Reiter (2011) showed that an attacker who knows the measurement
Jacobian ``H`` can craft an attack ``a = H c`` that shifts the estimated
state by ``c`` while leaving the residual unchanged — so it is *stealthy*
against the chi-square bad-data detector.  This module builds such attacks
(stealthy, partial-knowledge and random), produces labelled datasets for
training detectors, and connects to moving-target defense: after an MTD
reconfiguration the attacker's assumed ``H`` is stale, and the once-stealthy
attack becomes detectable.

These tools are for defensive research — training and evaluating detectors
and defenses on simulated grids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .estimation import DCStateEstimator


@dataclass
class AttackDataset:
    """Labelled measurement dataset for detector training.

    Attributes
    ----------
    Z : ndarray ``(n, m)``
        Measurement vectors (clean and attacked).
    y : ndarray ``(n,)``
        Labels (0 = clean, 1 = attacked).
    residual_norm : ndarray ``(n,)``
        Chi-square statistic of each sample (useful as a baseline feature).
    delta : ndarray ``(n, m)``
        Injected attack vector (zeros for clean samples).
    """

    Z: np.ndarray
    y: np.ndarray
    residual_norm: np.ndarray
    delta: np.ndarray
    meta: Dict = None

    def split(self, test: float = 0.3, seed: int = 0):
        """Stratified train/test split."""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.y))
        k = int(len(idx) * test)
        te, tr = idx[:k], idx[k:]
        return (self.Z[tr], self.y[tr]), (self.Z[te], self.y[te])


class FDIAttacker:
    """Generate false-data-injection attacks against a DC estimator.

    Parameters
    ----------
    estimator:
        A :class:`~tavan.power.DCStateEstimator` (defines ``H`` and the detector).
    knowledge:
        Fraction of meters the attacker can read/alter (1.0 = full knowledge).
    seed:
        Random seed.
    """

    def __init__(self, estimator: DCStateEstimator, knowledge: float = 1.0, seed: Optional[int] = None) -> None:
        self.est = estimator
        self.H = estimator.H
        self.m, self.n = self.H.shape
        self.knowledge = knowledge
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------- attack vectors
    def stealthy_attack(self, target_state: Optional[np.ndarray] = None, magnitude: float = 0.1,
                        targets: Optional[Sequence[int]] = None) -> np.ndarray:
        """Construct a stealthy attack ``a = H c``.

        ``c`` is the intended shift of the state estimate (non-slack angles);
        if not given, a random ``c`` of the requested ``magnitude`` is used,
        optionally restricted to ``targets`` state variables.
        """
        if target_state is None:
            c = np.zeros(self.n)
            idx = targets if targets is not None else self.rng.choice(self.n, max(1, self.n // 5), replace=False)
            c[idx] = self.rng.normal(0, magnitude, len(idx))
        else:
            c = np.asarray(target_state, dtype=float)
        return self.H @ c

    def partial_stealthy_attack(self, magnitude: float = 0.1) -> np.ndarray:
        """Stealthy attack limited to the meters the attacker controls.

        With partial knowledge the attacker can only change a subset of
        meters; the attack is projected onto the controllable set, so it is
        no longer perfectly stealthy — a realistic, detectable-with-some-
        probability threat.
        """
        n_ctrl = max(1, int(self.knowledge * self.m))
        ctrl = np.sort(self.rng.choice(self.m, n_ctrl, replace=False))
        a = self.stealthy_attack(magnitude=magnitude)
        mask = np.zeros(self.m, dtype=bool)
        mask[ctrl] = True
        a[~mask] = 0.0
        return a

    def random_attack(self, magnitude: float = 0.1) -> np.ndarray:
        """Naive random attack on controlled meters (easily detected)."""
        n_ctrl = max(1, int(self.knowledge * self.m))
        ctrl = self.rng.choice(self.m, n_ctrl, replace=False)
        a = np.zeros(self.m)
        a[ctrl] = self.rng.normal(0, magnitude * (np.abs(self.H).mean() + 1e-9) * 5, len(ctrl))
        return a

    def is_stealthy(self, clean_z: np.ndarray, attack: np.ndarray) -> bool:
        """Does ``clean_z + attack`` pass the chi-square detector?"""
        alarm, _ = self.est.detect(clean_z + attack)
        return not bool(np.atleast_1d(alarm)[0])

    # ----------------------------------------------------------- dataset
    def make_dataset(self, injections: np.ndarray, attack_ratio: float = 0.5, magnitude: float = 0.1,
                     attack_type: str = "stealthy", noise_seed: Optional[int] = None) -> AttackDataset:
        """Build a labelled detector-training dataset from operating points.

        Parameters
        ----------
        injections:
            Nodal injection scenarios ``(n, n_bus)`` (e.g. from a
            :class:`~tavan.generate.ScenarioGenerator`).
        attack_ratio:
            Fraction of samples that are attacked.
        magnitude:
            Attack strength (state shift for stealthy attacks).
        attack_type:
            ``"stealthy"``, ``"partial"`` or ``"random"``.
        """
        injections = np.atleast_2d(injections)
        Z = self.est.simulate_measurements(injections, seed=noise_seed)
        Z = np.atleast_2d(Z)
        n = len(Z)
        y = (self.rng.random(n) < attack_ratio).astype(int)
        delta = np.zeros_like(Z)
        builder = {"stealthy": lambda: self.stealthy_attack(magnitude=magnitude),
                   "partial": lambda: self.partial_stealthy_attack(magnitude),
                   "random": lambda: self.random_attack(magnitude)}[attack_type]
        for i in np.where(y == 1)[0]:
            delta[i] = builder()
        Zc = Z + delta
        _, p = self.est.detect(Zc)
        J = self.est.estimate(Zc)
        rn = np.einsum("ni,ij,nj->n", Zc - J.estimate, self.est.R_inv, Zc - J.estimate)
        return AttackDataset(Zc, y, rn, delta,
                             meta={"attack_type": attack_type, "magnitude": magnitude,
                                   "knowledge": self.knowledge, "threshold": self.est.threshold})

    def evaluate_defense(self, injections: np.ndarray, defended_estimator: DCStateEstimator,
                         magnitude: float = 0.1, n: Optional[int] = None) -> Dict[str, float]:
        """Detection rate of attacks (built against ``self``) under a defended grid.

        The attacker crafts stealthy attacks assuming the *original* ``H``;
        ``defended_estimator`` uses the post-MTD topology.  Returns the
        fraction of these attacks the defended detector catches — the key
        metric in moving-target-defense studies.
        """
        injections = np.atleast_2d(injections)
        if n:
            injections = injections[:n]
        caught = 0
        total = len(injections)
        for p in injections:
            z = defended_estimator.simulate_measurements(p, seed=int(self.rng.integers(2 ** 31)))
            a = self.stealthy_attack(magnitude=magnitude)
            alarm, _ = defended_estimator.detect(z + a)
            caught += int(bool(np.atleast_1d(alarm)[0]))
        return {"detection_rate": caught / max(total, 1), "n": total, "magnitude": magnitude}

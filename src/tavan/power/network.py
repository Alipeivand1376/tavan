"""Power-network model and DC power flow.

A minimal, dependency-free representation of a transmission network built
from bus and branch tables, with the DC power-flow model used for linear
state estimation and false-data-injection studies.

Conventions follow MATPOWER: buses carry a type (1 = PQ, 2 = PV, 3 = slack),
branches have per-unit series reactance ``x``, and power is in MW on a
``base_mva`` base.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PowerNetwork:
    """A transmission network for DC analysis.

    Parameters
    ----------
    n_bus:
        Number of buses (0-indexed internally).
    branches:
        List of ``(from_bus, to_bus, reactance_pu)`` triples.
    slack:
        Index of the reference (slack) bus.
    base_mva:
        Power base (MVA).
    bus_names, branch_ratings:
        Optional metadata.
    """

    n_bus: int
    branches: List[Tuple[int, int, float]]
    slack: int = 0
    base_mva: float = 100.0
    bus_names: Optional[List[str]] = None
    branch_ratings: Optional[np.ndarray] = None

    def __post_init__(self):
        self.branches = [(int(f), int(t), float(x)) for f, t, x in self.branches]
        self.n_branch = len(self.branches)
        self._build()

    # ------------------------------------------------------------ build
    def _build(self):
        nb, nl = self.n_bus, self.n_branch
        A = np.zeros((nl, nb))           # branch-bus incidence
        b = np.zeros(nl)                  # series susceptance (1/x)
        for k, (f, t, x) in enumerate(self.branches):
            A[k, f] = 1.0
            A[k, t] = -1.0
            b[k] = 1.0 / x
        self.incidence = A
        self.b_branch = b
        Bf = (b[:, None] * A)            # branch flow matrix (per-unit angle -> flow)
        self.Bf = Bf
        self.Bbus = A.T @ Bf             # nodal susceptance matrix
        self._noslack = np.array([i for i in range(nb) if i != self.slack])

    # --------------------------------------------------------- power flow
    def dc_power_flow(self, p_inject: np.ndarray) -> Dict[str, np.ndarray]:
        """Solve DC power flow for nodal injections ``p_inject`` (MW).

        Returns a dict with bus voltage angles ``theta`` (rad) and branch
        active-power flows ``flow`` (MW).
        """
        p = np.asarray(p_inject, dtype=float) / self.base_mva
        if p.shape[-1] != self.n_bus:
            raise ValueError(f"p_inject must have {self.n_bus} entries")
        ns = self._noslack
        theta = np.zeros_like(p)
        Bred = self.Bbus[np.ix_(ns, ns)]
        theta_ns = np.linalg.solve(Bred, np.take(p, ns, axis=-1).T).T
        if p.ndim == 1:
            theta[ns] = theta_ns
            flow = self.Bf @ theta * self.base_mva
        else:
            theta[:, ns] = theta_ns
            flow = theta @ self.Bf.T * self.base_mva
        return {"theta": theta, "flow": flow}

    # ---------------------------------------------- measurement model
    def measurement_matrix(self, measurements: Sequence[str] = ("flow", "injection")) -> Tuple[np.ndarray, List[str]]:
        """DC measurement Jacobian ``H`` mapping non-slack angles to measurements.

        ``measurements`` selects which quantities are metered: ``"flow"``
        (all branch flows), ``"injection"`` (all non-slack bus injections).
        Returns ``H`` and a list of measurement labels.
        """
        ns = self._noslack
        rows, labels = [], []
        if "flow" in measurements:
            Hf = self.Bf[:, ns] * self.base_mva
            rows.append(Hf)
            labels += [f"flow[{f}->{t}]" for f, t, _ in self.branches]
        if "injection" in measurements:
            Hi = self.Bbus[np.ix_(ns, ns)] * self.base_mva
            rows.append(Hi)
            labels += [f"inj[{i}]" for i in ns]
        H = np.vstack(rows) if rows else np.zeros((0, len(ns)))
        return H, labels

    def topology_hash(self) -> int:
        """Hash of the branch set (changes when MTD reconfigures the grid)."""
        return hash(tuple(sorted((f, t, round(x, 9)) for f, t, x in self.branches)))

    def perturb_reactances(self, factors, seed: Optional[int] = None) -> "PowerNetwork":
        """Return a copy with branch reactances scaled — a moving-target-defense move.

        ``factors`` is a scalar range (reactances scaled by U[1-f, 1+f]) or an
        explicit per-branch array (e.g. from D-FACTS devices).
        """
        rng = np.random.default_rng(seed)
        if np.isscalar(factors):
            scale = rng.uniform(1 - factors, 1 + factors, self.n_branch)
        else:
            scale = np.asarray(factors, dtype=float)
        new = [(f, t, x * s) for (f, t, x), s in zip(self.branches, scale)]
        return PowerNetwork(self.n_bus, new, self.slack, self.base_mva, self.bus_names, self.branch_ratings)

    # --------------------------------------------------------------- I/O
    @classmethod
    def from_matpower(cls, bus: np.ndarray, branch: np.ndarray, base_mva: float = 100.0) -> "PowerNetwork":
        """Build from MATPOWER-style ``bus`` and ``branch`` matrices.

        ``bus[:, 0]`` bus ids, ``bus[:, 1]`` types (3 = slack); ``branch[:, 0:2]``
        from/to ids, ``branch[:, 3]`` reactance, optional ``branch[:, 5]`` rating.
        """
        bus = np.asarray(bus, dtype=float)
        branch = np.asarray(branch, dtype=float)
        ids = bus[:, 0].astype(int)
        remap = {bid: i for i, bid in enumerate(ids)}
        slack = int(np.where(bus[:, 1] == 3)[0][0]) if np.any(bus[:, 1] == 3) else 0
        branches = [(remap[int(r[0])], remap[int(r[1])], float(r[3])) for r in branch]
        ratings = branch[:, 5] if branch.shape[1] > 5 and np.any(branch[:, 5] > 0) else None
        return cls(len(ids), branches, slack, base_mva, [str(i) for i in ids], ratings)

    def summary(self) -> dict:
        return {"n_bus": self.n_bus, "n_branch": self.n_branch, "slack": self.slack,
                "base_mva": self.base_mva}

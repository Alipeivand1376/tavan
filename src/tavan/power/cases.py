"""Standard IEEE test cases (bundled, no downloads).

Branch reactances are the standard IEEE values in per unit on a 100 MVA
base.  Bus ids are 0-indexed here.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .network import PowerNetwork

# IEEE 14-bus branch data: (from, to, x_pu), bus ids 1..14 -> shifted to 0..13
_IEEE14_BRANCHES_1 = [
    (1, 2, 0.05917), (1, 5, 0.22304), (2, 3, 0.19797), (2, 4, 0.17632), (2, 5, 0.17388),
    (3, 4, 0.17103), (4, 5, 0.04211), (4, 7, 0.20912), (4, 9, 0.55618), (5, 6, 0.25202),
    (6, 11, 0.19890), (6, 12, 0.25581), (6, 13, 0.13027), (7, 8, 0.17615), (7, 9, 0.11001),
    (9, 10, 0.08450), (9, 14, 0.27038), (10, 11, 0.19207), (12, 13, 0.19988), (13, 14, 0.34802),
]

# IEEE 9-bus (WSCC) branch data
_IEEE9_BRANCHES_1 = [
    (1, 4, 0.0576), (4, 5, 0.092), (5, 6, 0.17), (3, 6, 0.0586), (6, 7, 0.1008),
    (7, 8, 0.072), (8, 2, 0.0625), (8, 9, 0.161), (9, 4, 0.085),
]

# IEEE 30-bus branch data (standard reactances)
_IEEE30_BRANCHES_1 = [
    (1, 2, 0.06), (1, 3, 0.19), (2, 4, 0.17), (3, 4, 0.037), (2, 5, 0.258), (2, 6, 0.197),
    (4, 6, 0.037), (5, 7, 0.116), (6, 7, 0.082), (6, 8, 0.042), (6, 9, 0.208), (6, 10, 0.556),
    (9, 11, 0.208), (9, 10, 0.11), (4, 12, 0.256), (12, 13, 0.14), (12, 14, 0.2559),
    (12, 15, 0.1304), (12, 16, 0.1987), (14, 15, 0.1997), (16, 17, 0.1932), (15, 18, 0.2185),
    (18, 19, 0.1292), (19, 20, 0.068), (10, 20, 0.209), (10, 17, 0.0845), (10, 21, 0.0749),
    (10, 22, 0.1499), (21, 22, 0.0236), (15, 23, 0.202), (22, 24, 0.179), (23, 24, 0.27),
    (24, 25, 0.3292), (25, 26, 0.38), (25, 27, 0.2087), (28, 27, 0.396), (27, 29, 0.4153),
    (27, 30, 0.6027), (29, 30, 0.4533), (8, 28, 0.2), (6, 28, 0.0599),
]


def _mk(branches_1: List[Tuple[int, int, float]], n_bus: int, slack: int = 0) -> PowerNetwork:
    branches = [(f - 1, t - 1, x) for f, t, x in branches_1]
    net = PowerNetwork(n_bus, branches, slack=slack, base_mva=100.0,
                       bus_names=[str(i + 1) for i in range(n_bus)])
    return net


def ieee9() -> PowerNetwork:
    """WSCC 9-bus system."""
    return _mk(_IEEE9_BRANCHES_1, 9)


def ieee14() -> PowerNetwork:
    """IEEE 14-bus system (slack at bus 1)."""
    return _mk(_IEEE14_BRANCHES_1, 14)


def ieee30() -> PowerNetwork:
    """IEEE 30-bus system."""
    return _mk(_IEEE30_BRANCHES_1, 30)


CASES = {"ieee9": ieee9, "ieee14": ieee14, "ieee30": ieee30}


def load_case(name: str) -> PowerNetwork:
    """Load a bundled test case by name (``"ieee9"``, ``"ieee14"``, ``"ieee30"``)."""
    key = name.lower().replace("case", "ieee").replace("_", "")
    if key not in CASES:
        raise ValueError(f"unknown case {name!r}; available: {sorted(CASES)}")
    return CASES[key]()


def random_injection(net: PowerNetwork, n: int = 1, load_scale: float = 50.0, seed=None) -> np.ndarray:
    """Random balanced nodal injections for a network (slack absorbs the mismatch)."""
    rng = np.random.default_rng(seed)
    p = rng.normal(0, load_scale, size=(n, net.n_bus))
    p[:, net.slack] = 0.0
    p[:, net.slack] = -p.sum(axis=1)  # balance at slack
    return p[0] if n == 1 else p

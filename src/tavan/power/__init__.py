"""Power-system models for DC analysis, state estimation and FDI studies.

* :class:`PowerNetwork` — buses/branches with DC power flow and the
  measurement Jacobian; :meth:`~PowerNetwork.perturb_reactances` implements a
  moving-target-defense reconfiguration.
* :class:`DCStateEstimator` — weighted least squares with a chi-square
  bad-data detector.
* :class:`FDIAttacker` — stealthy / partial / random false-data-injection
  attacks and labelled datasets for training detectors (defensive research).
* Solar geometry and clear-sky irradiance for PV envelopes.
* Bundled IEEE test cases (9, 14, 30 bus).
"""

from .cases import ieee9, ieee14, ieee30, load_case, random_injection
from .estimation import DCStateEstimator, EstimationResult
from .fdi import AttackDataset, FDIAttacker
from .network import PowerNetwork
from .solar import clear_sky_ghi, clear_sky_pv, daylight_mask, solar_zenith

__all__ = [
    "PowerNetwork", "DCStateEstimator", "EstimationResult", "FDIAttacker", "AttackDataset",
    "ieee9", "ieee14", "ieee30", "load_case", "random_injection",
    "clear_sky_pv", "clear_sky_ghi", "solar_zenith", "daylight_mask",
]

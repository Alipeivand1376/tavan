"""Generative models for time-series scenarios.

Low-level generators (``fit(X, cond)`` / ``sample(n, cond)``):

=================  ==========================================================
``bootstrap``      analog ensemble / bootstrap resampling (baseline)
``copula``         Gaussian copula with empirical marginals
``gmm``            Gaussian mixture (EM, BIC model selection, likelihoods)
``vae``            conditional variational auto-encoder
``diffusion``      DDPM training + DDIM sampling, classifier-free guidance
``flow``           RealNVP normalising flow (exact likelihoods)
``gan``            conditional GAN with moment matching
=================  ==========================================================

The high-level :class:`ScenarioGenerator` wraps preprocessing, conditioning
and physical constraints around any of them.
"""

from .base import BaseGenerator, NeuralGenerator
from .conditions import encode_calendar
from .constraints import (
    Bounds,
    ClearSkyCap,
    Constraint,
    Custom,
    EnergyBudget,
    EnvelopeCap,
    NonNegative,
    RampLimit,
    apply_constraints,
    violation_rates,
)
from .diffusion import Diffusion
from .flow import RealNVP
from .gan import CGAN
from .pipeline import ScenarioGenerator
from .statistical import Bootstrap, GaussianCopula, GaussianMixture
from .vae import VAE

__all__ = [
    "BaseGenerator", "NeuralGenerator", "Bootstrap", "GaussianCopula", "GaussianMixture", "VAE",
    "Diffusion", "RealNVP", "CGAN", "ScenarioGenerator", "Constraint", "Bounds", "NonNegative",
    "RampLimit", "EnvelopeCap", "ClearSkyCap", "EnergyBudget", "Custom", "apply_constraints",
    "violation_rates", "encode_calendar",
]

"""Base classes for generative models.

Low-level generators operate on 2-D arrays ``X`` of shape
``(n_samples, n_features)`` — typically flattened, scaled daily profiles —
with optional conditions ``cond`` of shape ``(n_samples, n_cond)``.  The
high-level :class:`~tavan.generate.ScenarioGenerator` takes care of
preprocessing and reshaping.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .._validation import check_finite, check_is_fitted, make_rng
from ..base import BaseEstimator
from ..config import TrainConfig


class BaseGenerator(BaseEstimator):
    """Interface of all generative models.

    Sub-classes implement :meth:`_fit` and :meth:`_sample`.
    """

    supports_conditions: bool = True
    supports_likelihood: bool = False

    def fit(self, X, cond=None) -> "BaseGenerator":
        """Fit the model to samples ``X`` ``(n, d)`` with optional ``cond`` ``(n, k)``."""
        X = check_finite(np.asarray(X, dtype=float), "X")
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n_samples, n_features), got {X.shape}")
        if X.shape[0] < 2:
            raise ValueError("at least two samples are required")
        cond = self._check_cond(cond, X.shape[0], fitting=True)
        self.n_features_ = X.shape[1]
        self.n_cond_ = 0 if cond is None else cond.shape[1]
        self._fit(X, cond)
        self.is_fitted_ = True
        return self

    def sample(self, n: int, cond=None, seed: Optional[int] = None) -> np.ndarray:
        """Draw ``n`` samples; ``cond`` is one row (broadcast) or ``n`` rows."""
        check_is_fitted(self)
        n = int(n)
        if n <= 0:
            raise ValueError("n must be positive")
        if cond is not None:
            cond = np.atleast_2d(np.asarray(cond, dtype=float))
            if cond.shape[0] == 1 and n > 1:
                cond = np.repeat(cond, n, axis=0)
            if cond.shape != (n, self.n_cond_):
                raise ValueError(f"cond must have shape (1 or {n}, {self.n_cond_}), got {cond.shape}")
        elif self.n_cond_:
            raise ValueError("this model was fitted with conditions; pass `cond` to sample()")
        rng = make_rng(seed if seed is not None else getattr(self, "seed", None))
        return self._sample(n, cond, rng)

    def partial_fit(self, X, cond=None, epochs: Optional[int] = None) -> "BaseGenerator":
        """Update the model with new data (default: refit on ``X``)."""
        return self.fit(X, cond)

    def score_samples(self, X, cond=None) -> np.ndarray:
        """Per-sample log-likelihood (only for likelihood-based models)."""
        raise NotImplementedError(f"{type(self).__name__} does not provide likelihoods")

    # --------------------------------------------------------- internals
    def _check_cond(self, cond, n: int, fitting: bool = False) -> Optional[np.ndarray]:
        if cond is None:
            return None
        if not self.supports_conditions:
            raise ValueError(f"{type(self).__name__} does not support conditions")
        cond = np.asarray(cond, dtype=float)
        if cond.ndim == 1:
            cond = cond[:, None]
        if cond.shape[0] != n:
            raise ValueError("cond must have one row per sample")
        return check_finite(cond, "cond")

    def _fit(self, X: np.ndarray, cond: Optional[np.ndarray]) -> None:  # pragma: no cover
        raise NotImplementedError

    def _sample(self, n: int, cond: Optional[np.ndarray], rng: np.random.Generator) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class NeuralGenerator(BaseGenerator):
    """Shared plumbing for generators built on :mod:`tavan.nn`.

    Sub-classes define ``_module_attrs`` (names of :class:`~tavan.nn.Module`
    attributes) and :meth:`_build`.
    """

    _module_attrs: Tuple[str, ...] = ("net_",)

    def _train_config(self, **override) -> TrainConfig:
        base = TrainConfig.from_any(getattr(self, "train", None))
        fields: Dict[str, Any] = {}
        for k in ("epochs", "lr", "batch_size", "seed", "verbose"):
            v = getattr(self, k, None)
            if v is not None:
                fields[k] = v
        fields.update({k: v for k, v in override.items() if v is not None})
        return base.update(**fields)

    def _build(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _get_state(self) -> Dict[str, Any]:
        st = {"n_features_": self.n_features_, "n_cond_": self.n_cond_, "is_fitted_": True}
        for a in self._module_attrs:
            st[a] = getattr(self, a)
        for a in self._state_attrs:
            st[a] = getattr(self, a)
        return st

    def _set_state(self, state: Dict[str, Any]) -> None:
        modules = {a: state.pop(a) for a in self._module_attrs if a in state}
        for k, v in state.items():
            setattr(self, k, v)
        self._build()
        for a, sd in modules.items():
            getattr(self, a).load_state_dict(sd)
            getattr(self, a).eval()

    @property
    def history(self):
        """Training history (:class:`tavan.nn.History`)."""
        return getattr(self, "history_", None)


def _nn_cond(X: np.ndarray, cond: Optional[np.ndarray]) -> np.ndarray:
    return X if cond is None else np.concatenate([X, cond], axis=1)

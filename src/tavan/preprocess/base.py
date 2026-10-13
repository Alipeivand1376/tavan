"""Base class for reversible transforms."""

from __future__ import annotations

import numpy as np

from ..base import BaseEstimator


class BaseTransform(BaseEstimator):
    """A fit/transform/inverse_transform step operating on ``(n, d)`` arrays.

    Inputs of shape ``(T,)`` are treated as one column; 3-D inputs
    ``(n, L, C)`` are handled per channel (the last axis).
    """

    def fit(self, x, y=None) -> "BaseTransform":
        return self

    def transform(self, x) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def inverse_transform(self, x) -> np.ndarray:
        return np.asarray(x, dtype=float)

    def fit_transform(self, x, y=None) -> np.ndarray:
        return self.fit(x, y).transform(x)


def _flatten_channels(x):
    """Return ``(x2d, restore)`` collapsing all but the last axis."""
    arr = np.asarray(x, dtype=float)
    shape = arr.shape
    flat = arr.reshape(-1, 1) if arr.ndim == 1 else arr.reshape(-1, shape[-1])
    return flat, lambda y: np.asarray(y).reshape(shape)

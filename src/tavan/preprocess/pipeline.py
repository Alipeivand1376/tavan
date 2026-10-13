"""Composable preprocessing pipelines."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..registry import register
from .base import BaseTransform


@register("transform", "pipeline")
class Pipeline(BaseTransform):
    """Apply transforms in order; ``inverse_transform`` undoes them in reverse.

    >>> from tavan.preprocess import Pipeline, OutlierCleaner, Imputer, StandardScaler
    >>> pipe = Pipeline([OutlierCleaner(), Imputer("seasonal", period=24), StandardScaler()])
    """

    _state_attrs = ()

    def __init__(self, steps: Sequence[BaseTransform] = ()) -> None:
        self.steps = list(steps)

    def fit(self, x, y=None) -> "Pipeline":
        self.fit_transform(x)
        return self

    def fit_transform(self, x, y=None) -> np.ndarray:
        z = np.asarray(x, dtype=float)
        for s in self.steps:
            z = s.fit_transform(z)
        return z

    def transform(self, x) -> np.ndarray:
        z = np.asarray(x, dtype=float)
        for s in self.steps:
            z = s.transform(z)
        return z

    def inverse_transform(self, x) -> np.ndarray:
        z = np.asarray(x, dtype=float)
        for s in reversed(self.steps):
            z = s.inverse_transform(z)
        return z

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, i) -> BaseTransform:
        return self.steps[i]


def build_pipeline(outliers=None, impute=None, scale=None, period=None) -> Pipeline:
    """Build the standard *clean → impute → scale* pipeline from short names.

    Each argument may be ``None`` (skip), a method name or a transform
    instance.
    """
    from .impute import Imputer
    from .outliers import OutlierCleaner
    from .scale import make_scaler

    steps: List[BaseTransform] = []
    if outliers:
        steps.append(outliers if isinstance(outliers, BaseTransform)
                     else OutlierCleaner(method="hampel" if outliers is True else outliers))
    if impute:
        steps.append(impute if isinstance(impute, BaseTransform)
                     else Imputer(method="auto" if impute is True else impute, period=period))
    elif outliers:
        steps.append(Imputer(method="auto", period=period))
    if scale:
        steps.append(make_scaler("standard" if scale is True else scale))
    return Pipeline(steps)

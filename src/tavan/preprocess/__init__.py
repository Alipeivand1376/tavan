"""Preprocessing: imputation, outlier cleaning, scaling and features.

Every step is an optional, reusable object with ``fit``/``transform``/
``inverse_transform`` and can be chained with :class:`Pipeline`.
"""

from .base import BaseTransform
from .features import calendar_features, day_type, fourier_terms, lag_matrix, one_hot, season
from .impute import Imputer, impute, missing_report
from .outliers import OutlierCleaner, detect_outliers
from .pipeline import Pipeline, build_pipeline
from .scale import IdentityScaler, LogScaler, MinMaxScaler, RobustScaler, StandardScaler, make_scaler

__all__ = [
    "BaseTransform", "Imputer", "impute", "missing_report", "OutlierCleaner", "detect_outliers",
    "StandardScaler", "MinMaxScaler", "RobustScaler", "LogScaler", "IdentityScaler", "make_scaler",
    "Pipeline", "build_pipeline", "calendar_features", "fourier_terms", "lag_matrix",
    "day_type", "season", "one_hot",
]

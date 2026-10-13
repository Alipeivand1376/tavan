"""The :class:`TimeSeries` container."""

from __future__ import annotations

import csv
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .._validation import as_2d
from .timeutils import date_range, infer_freq, parse_freq, steps_per, to_datetime64
from .windows import sliding_windows, to_profiles


class TimeSeries:
    """A regularly sampled, possibly multivariate time series.

    The container keeps values, timestamps and the chain of fitted
    transforms applied through its methods, so that model outputs can be
    mapped back to physical units with :meth:`inverse_transform`.

    Parameters
    ----------
    values:
        ``(T,)`` or ``(T, C)`` array.  ``NaN`` marks missing data.
    timestamps:
        Optional array of datetimes.  If omitted but ``start`` and ``freq``
        are given, a regular grid is generated.
    freq:
        Sampling interval such as ``"1h"`` or ``"15min"``; inferred from
        timestamps when possible.
    names:
        Channel names.
    unit:
        Free-text physical unit (used in plots and reports).

    Examples
    --------
    >>> import numpy as np
    >>> from tavan import TimeSeries
    >>> ts = TimeSeries(np.random.rand(48), start="2025-01-01", freq="1h")
    >>> ts.to_profiles(24).shape
    (2, 24, 1)
    """

    def __init__(self, values, timestamps=None, freq=None, start=None,
                 names: Optional[Sequence[str]] = None, unit: str = "") -> None:
        arr, self._was_1d = as_2d(values, "values")
        self.values = arr.astype(float)
        T, C = self.values.shape
        if timestamps is None and start is not None:
            if freq is None:
                raise ValueError("`freq` is required when `start` is given")
            timestamps = date_range(start, T, freq)
        self.timestamps = None if timestamps is None else to_datetime64(timestamps)
        if self.timestamps is not None and len(self.timestamps) != T:
            raise ValueError("timestamps and values must have the same length")
        if freq is None and self.timestamps is not None:
            freq = infer_freq(self.timestamps)
        self.freq = parse_freq(freq) if freq is not None else None
        self.names = list(names) if names is not None else [f"x{i}" for i in range(C)]
        if len(self.names) != C:
            raise ValueError("number of names must match number of channels")
        self.unit = unit
        self.transforms: List = []

    # ------------------------------------------------------------ basics
    @property
    def n_steps(self) -> int:
        return self.values.shape[0]

    @property
    def n_channels(self) -> int:
        return self.values.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.values.shape

    @property
    def steps_per_day(self) -> Optional[int]:
        return steps_per(self.freq, "1D") if self.freq is not None else None

    def __len__(self) -> int:
        return self.n_steps

    def __repr__(self) -> str:
        span = ""
        if self.timestamps is not None:
            span = f", {self.timestamps[0]} → {self.timestamps[-1]}"
        miss = self.missing_ratio()
        return (f"TimeSeries(n_steps={self.n_steps}, channels={self.names}, freq={self.freq}"
                f"{span}, missing={miss:.1%})")

    def __getitem__(self, idx) -> "TimeSeries":
        if isinstance(idx, str):
            c = self.names.index(idx)
            return self._new(self.values[:, [c]], names=[idx])
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        ts = self.timestamps[idx] if self.timestamps is not None else None
        out = self._new(self.values[idx], timestamps=ts)
        return out

    def _new(self, values, timestamps="same", names=None) -> "TimeSeries":
        ts = self.timestamps if isinstance(timestamps, str) and timestamps == "same" else timestamps
        out = TimeSeries(values, timestamps=ts, freq=self.freq, names=names or self.names, unit=self.unit)
        out._was_1d = self._was_1d and out.n_channels == 1
        out.transforms = list(self.transforms)
        return out

    def copy(self) -> "TimeSeries":
        return self._new(self.values.copy())

    def to_numpy(self, squeeze: bool = True) -> np.ndarray:
        """Values as an array; 1-channel series are squeezed to ``(T,)``."""
        if squeeze and self.n_channels == 1:
            return self.values[:, 0].copy()
        return self.values.copy()

    def __array__(self, dtype=None, copy=None):
        arr = self.to_numpy()
        return arr.astype(dtype) if dtype is not None else arr

    # -------------------------------------------------------- construction
    @classmethod
    def from_csv(cls, path: str, time_col: Union[str, int, None] = 0,
                 value_cols: Optional[Sequence[Union[str, int]]] = None, delimiter: str = ",",
                 freq=None, unit: str = "", missing: Sequence[str] = ("", "NA", "NaN", "nan", "null")) -> "TimeSeries":
        """Read a CSV file with the standard library (no pandas needed)."""
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh, delimiter=delimiter))
        header, body = rows[0], [r for r in rows[1:] if r]

        def col_index(c):
            return header.index(c) if isinstance(c, str) else int(c)

        ti = col_index(time_col) if time_col is not None else None
        if value_cols is None:
            vidx = [i for i in range(len(header)) if i != ti]
        else:
            vidx = [col_index(c) for c in value_cols]
        miss = set(missing)
        vals = np.array([[float(r[i]) if r[i].strip() not in miss else np.nan for i in vidx] for r in body])
        ts = [r[ti] for r in body] if ti is not None else None
        return cls(vals, timestamps=ts, freq=freq, names=[header[i] for i in vidx], unit=unit)

    @classmethod
    def from_pandas(cls, obj, freq=None, unit: str = "") -> "TimeSeries":
        """Create from a pandas ``Series``/``DataFrame`` with a DatetimeIndex."""
        idx = getattr(obj, "index", None)
        ts = None
        if idx is not None and hasattr(idx, "dtype") and "datetime" in str(idx.dtype):
            ts = np.asarray(idx.to_numpy(), dtype="datetime64[s]")
        names = list(obj.columns) if hasattr(obj, "columns") else [getattr(obj, "name", None) or "x0"]
        return cls(obj.to_numpy(dtype=float), timestamps=ts, freq=freq, names=[str(n) for n in names], unit=unit)

    def to_pandas(self):
        """Convert to a ``pandas.DataFrame`` (requires pandas)."""
        import pandas as pd

        index = pd.DatetimeIndex(self.timestamps) if self.timestamps is not None else None
        return pd.DataFrame(self.values, index=index, columns=self.names)

    def to_csv(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow((["timestamp"] if self.timestamps is not None else []) + self.names)
            for i in range(self.n_steps):
                row = [f"{v:.10g}" if np.isfinite(v) else "" for v in self.values[i]]
                w.writerow(([str(self.timestamps[i])] if self.timestamps is not None else []) + row)

    # ------------------------------------------------------ missing data
    def missing_mask(self) -> np.ndarray:
        return ~np.isfinite(self.values)

    def missing_ratio(self) -> float:
        return float(self.missing_mask().mean()) if self.values.size else 0.0

    def missing_report(self) -> dict:
        from ..preprocess.impute import missing_report

        rep = missing_report(self.values)
        rep["names"] = self.names
        return rep

    # ------------------------------------------------------ preprocessing
    def _default_period(self, period):
        if period is not None:
            return period
        return self.steps_per_day

    def impute(self, method: str = "auto", period: Optional[int] = None, **kwargs) -> "TimeSeries":
        """Return a copy with missing values filled (see :class:`~tavan.preprocess.Imputer`)."""
        from ..preprocess.impute import Imputer

        imp = Imputer(method=method, period=self._default_period(period), **kwargs)
        out = self._new(imp.fit_transform(self.values))
        out.imputed_mask_ = imp.mask_
        return out

    def clean_outliers(self, method: str = "hampel", threshold: float = 3.5, window: Optional[int] = None,
                       action: str = "nan") -> "TimeSeries":
        """Return a copy with outliers replaced (``NaN`` by default, then impute)."""
        from ..preprocess.outliers import OutlierCleaner

        w = window or max(7, (self.steps_per_day or 24) + 1)
        oc = OutlierCleaner(method=method, threshold=threshold, window=w, action=action)
        out = self._new(oc.transform(self.values))
        out.outlier_mask_ = oc.mask_
        return out

    def scale(self, method="standard") -> "TimeSeries":
        """Return a scaled copy; the fitted scaler is appended to ``transforms``."""
        from ..preprocess.scale import make_scaler

        sc = make_scaler(method)
        out = self._new(sc.fit_transform(self.values))
        out.transforms.append(sc)
        return out

    def inverse_transform(self, values) -> np.ndarray:
        """Undo :meth:`scale` (and other recorded transforms) on ``values``."""
        z = np.asarray(values, dtype=float)
        for t in reversed(self.transforms):
            z = t.inverse_transform(z)
        return z

    def resample(self, factor: int, agg: str = "mean") -> "TimeSeries":
        """Down-sample by an integer ``factor`` (e.g. 15-min → hourly: 4)."""
        T = (self.n_steps // factor) * factor
        blocks = self.values[:T].reshape(-1, factor, self.n_channels)
        fn = {"mean": np.nanmean, "sum": np.nansum, "max": np.nanmax, "min": np.nanmin,
              "first": lambda a, axis: a.take(0, axis=axis), "last": lambda a, axis: a.take(-1, axis=axis)}[agg]
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vals = fn(blocks, axis=1)
        ts = self.timestamps[:T:factor] if self.timestamps is not None else None
        out = TimeSeries(vals, timestamps=ts, freq=self.freq * factor if self.freq is not None else None,
                         names=self.names, unit=self.unit)
        out.transforms = list(self.transforms)
        return out

    # -------------------------------------------------------- reshaping
    def to_profiles(self, period: Optional[int] = None, drop_incomplete: bool = True) -> np.ndarray:
        """``(n_periods, period, C)`` array, e.g. one row per day."""
        p = self._default_period(period)
        if p is None:
            raise ValueError("period is required (cannot infer it without freq)")
        return to_profiles(self.values, p, drop_incomplete=drop_incomplete)

    def profile_timestamps(self, period: Optional[int] = None) -> Optional[np.ndarray]:
        """Start timestamp of every complete period."""
        if self.timestamps is None:
            return None
        p = self._default_period(period)
        n = self.n_steps // p
        return self.timestamps[: n * p: p]

    def windows(self, lookback: int, horizon: int, stride: int = 1):
        """Supervised windows ``(X, Y)``; see :func:`tavan.data.sliding_windows`."""
        return sliding_windows(self.values, lookback, horizon, stride)

    def split(self, test: Union[float, int] = 0.2) -> Tuple["TimeSeries", "TimeSeries"]:
        """Chronological train/test split."""
        k = int(round(self.n_steps * test)) if isinstance(test, float) and test < 1 else int(test)
        return self[: self.n_steps - k], self[self.n_steps - k:]

    def calendar_features(self, features=("hour", "dow", "month"), cyclical: bool = True):
        """Calendar covariates; see :func:`tavan.preprocess.calendar_features`."""
        if self.timestamps is None:
            raise ValueError("timestamps are required for calendar features")
        from ..preprocess.features import calendar_features

        return calendar_features(self.timestamps, features, cyclical)

    def describe(self) -> dict:
        """Basic statistics per channel."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            v = self.values
            return {
                n: {
                    "mean": float(np.nanmean(v[:, i])), "std": float(np.nanstd(v[:, i])),
                    "min": float(np.nanmin(v[:, i])), "max": float(np.nanmax(v[:, i])),
                    "missing": float(np.mean(~np.isfinite(v[:, i]))),
                }
                for i, n in enumerate(self.names)
            }

    # ------------------------------------------------------------ plots
    def plot(self, ax=None, **kwargs):
        """Line plot with missing data highlighted (requires matplotlib)."""
        from ..viz import plot_series

        return plot_series(self, ax=ax, **kwargs)


def as_timeseries(x, freq=None, timestamps=None) -> TimeSeries:
    """Coerce arrays, pandas objects or TimeSeries to :class:`TimeSeries`."""
    if isinstance(x, TimeSeries):
        return x
    if hasattr(x, "to_numpy") and hasattr(x, "index"):
        return TimeSeries.from_pandas(x, freq=freq)
    return TimeSeries(x, timestamps=timestamps, freq=freq)

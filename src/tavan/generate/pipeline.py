"""End-to-end scenario generation: preprocess → fit → sample → constrain."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from .._validation import check_is_fitted, make_rng
from ..base import BaseEstimator
from ..data.timeseries import as_timeseries
from ..data.timeutils import to_datetime64
from ..data.windows import to_profiles
from ..preprocess.impute import Imputer
from ..preprocess.pipeline import build_pipeline
from ..preprocess.scale import StandardScaler, make_scaler
from ..registry import get, register
from ..scenarios.drift import DriftDetector, DriftResult
from ..scenarios.risk import scenario_loss, stratified_tail_sample
from ..scenarios.scenario_set import ScenarioSet
from .base import BaseGenerator
from .conditions import aggregate_to_profiles, date_steps, encode_calendar, parse_conditions
from .constraints import Constraint, apply_constraints, violation_rates


@register("pipeline", "scenario_generator")
class ScenarioGenerator(BaseEstimator):
    """Fit a generative model to a time series and produce scenarios.

    The user supplies raw data; cleaning, imputation, scaling, profile
    extraction, conditioning, training, inverse scaling and physical
    constraints are handled automatically.  Every step can be customised or
    disabled.

    Parameters
    ----------
    model:
        Generator name (``"diffusion"``, ``"vae"``, ``"flow"``, ``"gan"``,
        ``"gmm"``, ``"copula"``, ``"bootstrap"``) or a generator instance.
    period:
        Profile length in steps (default: one day, inferred from ``freq``).
    outliers:
        ``None``/``False`` (skip), ``True``/``"hampel"``, ``"zscore"``,
        ``"iqr"`` or an :class:`~tavan.preprocess.OutlierCleaner`.
    impute:
        Imputation method name (see :class:`~tavan.preprocess.Imputer`) or
        instance.  ``None`` requires complete data.
    scale:
        ``"standard"``, ``"robust"``, ``"minmax"``, ``"log"`` or ``None``.
    conditions:
        ``None`` (unconditional), a list from ``("dow", "weekend", "month",
        "season", "doy")``, ``"calendar"`` (weekend + day-of-year), and/or
        ``"custom"`` to use the ``cond`` array passed to :meth:`fit`.
    weekend:
        Weekend days for the ``weekend`` condition (Monday = 0).
    constraints:
        List of :class:`~tavan.generate.constraints.Constraint` objects
        applied to generated scenarios in physical units.
    model_params:
        Keyword arguments for the generator (e.g. ``{"epochs": 300}``).
    seed:
        Random seed for training and sampling.
    verbose:
        Print training progress.

    Examples
    --------
    >>> from tavan.data import make_load
    >>> from tavan.generate import ScenarioGenerator
    >>> gen = ScenarioGenerator("gmm", conditions=["weekend"]).fit(make_load(120))
    >>> scen = gen.sample(200, date="2024-06-01")
    >>> scen.values.shape
    (200, 24, 1)
    """

    _state_attrs = ("model_", "pre_", "scaler_", "cond_scaler_", "X_", "C_", "reference_", "period_",
                    "n_channels_", "names_", "unit_", "freq_seconds_", "cal_names_", "use_custom_",
                    "n_custom_", "is_fitted_")

    def __init__(self, model: Union[str, BaseGenerator] = "diffusion", period: Optional[int] = None,
                 outliers=None, impute="auto", scale="standard", conditions=None,
                 weekend: Sequence[int] = (5, 6), constraints: Optional[Sequence[Constraint]] = None,
                 model_params: Optional[Dict[str, Any]] = None, seed: Optional[int] = 0,
                 verbose: int = 0) -> None:
        self.model = model
        self.period = period
        self.outliers = outliers
        self.impute = impute
        self.scale = scale
        self.conditions = conditions
        self.weekend = weekend
        self.constraints = constraints
        self.model_params = model_params
        self.seed = seed
        self.verbose = verbose

    # ----------------------------------------------------------- helpers
    def _make_model(self) -> BaseGenerator:
        if isinstance(self.model, BaseGenerator):
            return self.model
        cls = get("generator", self.model)
        params = dict(self.model_params or {})
        names = cls._param_names()
        if "seed" in names and "seed" not in params:
            params["seed"] = self.seed
        if "verbose" in names and "verbose" not in params:
            params["verbose"] = self.verbose
        return cls(**params)

    def _profiles(self, data, cond=None, timestamps=None, freq=None, fit: bool = False):
        """Return physical profiles (n, P, C), starts (n,) or None, raw custom cond."""
        if isinstance(data, np.ndarray) and data.ndim == 3:
            prof = data.astype(float)
            if np.isnan(prof).any():
                prof = np.stack([Imputer("linear").fit_transform(p) for p in prof])
            if fit:
                self.period_ = prof.shape[1]
                self.n_channels_ = prof.shape[2]
                self.names_ = [f"x{i}" for i in range(prof.shape[2])]
                self.unit_ = ""
                self.freq_seconds_ = None
                self.pre_ = None
            starts = None if timestamps is None else to_datetime64(timestamps)
            return prof, starts, cond
        ts = as_timeseries(data, freq=freq, timestamps=timestamps)
        if fit:
            period = self.period or ts.steps_per_day
            if period is None:
                raise ValueError("cannot infer `period`; pass period=... or data with timestamps/freq")
            self.period_ = int(period)
            self.n_channels_ = ts.n_channels
            self.names_ = list(ts.names)
            self.unit_ = ts.unit
            self.freq_seconds_ = None if ts.freq is None else int(ts.freq / np.timedelta64(1, "s"))
            self.pre_ = build_pipeline(self.outliers, self.impute, None, self.period_)
            values = self.pre_.fit_transform(ts.values)
        else:
            values = self.pre_.transform(ts.values) if self.pre_ is not None else ts.values
        if np.isnan(values).any():
            raise ValueError("data contain NaN; enable `impute` (e.g. impute='seasonal')")
        prof = to_profiles(values, self.period_)
        starts = ts.profile_timestamps(self.period_)
        return prof, starts, cond

    def _cond_matrix(self, starts, cond, n: int, fit: bool = False) -> Optional[np.ndarray]:
        parts = []
        if self.cal_names_:
            if starts is None:
                raise ValueError("calendar conditions need timestamps (pass a TimeSeries or `timestamps`)")
            parts.append(encode_calendar(starts, self.cal_names_, self.weekend)[0])
        if self.use_custom_:
            if cond is None:
                raise ValueError("conditions include 'custom': pass `cond`")
            c = aggregate_to_profiles(cond, self.period_, n)
            if fit:
                self.cond_scaler_ = StandardScaler().fit(c)
                self.n_custom_ = c.shape[1]
            parts.append(self.cond_scaler_.transform(c))
        return np.concatenate(parts, axis=1) if parts else None

    def _flatten(self, prof_scaled: np.ndarray) -> np.ndarray:
        return prof_scaled.reshape(prof_scaled.shape[0], -1)

    def _unflatten(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], self.period_, self.n_channels_)

    # --------------------------------------------------------------- fit
    def fit(self, data, cond=None, timestamps=None, freq=None) -> "ScenarioGenerator":
        """Fit on a series (``TimeSeries``, ``(T,)``/``(T, C)`` array, pandas) or
        on ready-made profiles ``(n, P, C)``.

        ``cond`` provides custom conditions, either per step ``(T, k)``
        (aggregated per profile by their mean) or per profile ``(n, k)``.
        """
        self.cal_names_, self.use_custom_ = parse_conditions(self.conditions)
        self.n_custom_ = 0
        self.cond_scaler_ = None
        prof, starts, cond = self._profiles(data, cond, timestamps, freq, fit=True)
        self.reference_ = prof
        self.scaler_ = make_scaler(self.scale).fit(prof.reshape(-1, self.n_channels_))
        X = self._flatten(self.scaler_.transform(prof))
        C = self._cond_matrix(starts, cond, len(prof), fit=True)
        self.model_ = self._make_model()
        self.model_.fit(X, C)
        self.X_, self.C_ = X, C
        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------ sample
    def _sample_cond(self, n: int, date, cond, rng=None) -> Optional[np.ndarray]:
        if not (self.cal_names_ or self.use_custom_):
            return None
        # marginal sampling: no date and no custom conditions -> bootstrap the
        # training conditions, so sample(n) draws from the calendar's own mix
        if date is None and (cond is None or not self.use_custom_) and self.C_ is not None:
            r = rng if rng is not None else make_rng(self.seed)
            idx = r.integers(0, len(self.C_), n)
            return self.C_[idx]
        parts = []
        if self.cal_names_:
            if date is None:
                raise ValueError("this generator is conditioned on the calendar; pass `date=`")
            dates = to_datetime64(np.atleast_1d(date))
            enc = encode_calendar(dates, self.cal_names_, self.weekend)[0]
            parts.append(np.repeat(enc, n, 0) if len(enc) == 1 else enc)
        if self.use_custom_:
            if cond is None:
                raise ValueError("this generator uses custom conditions; pass `cond=`")
            c = np.atleast_2d(np.asarray(cond, dtype=float))
            if c.shape[1] != self.n_custom_ and c.shape[0] == 1:
                c = c.reshape(1, -1)
            c = self.cond_scaler_.transform(c)
            parts.append(np.repeat(c, n, 0) if len(c) == 1 else c)
        if not parts:
            return None
        out = np.concatenate(parts, axis=1)
        if len(out) != n:
            raise ValueError("per-scenario dates/conditions must have length n")
        return out

    def _context(self, date, n: int) -> Dict[str, Any]:
        if date is None or self.freq_seconds_ is None:
            return {"timestamps": None}
        dates = to_datetime64(np.atleast_1d(date))
        step = np.timedelta64(self.freq_seconds_, "s")
        steps = np.stack([date_steps(d, self.period_, step) for d in dates])
        if len(steps) == 1:
            steps = np.repeat(steps, n, 0)
        return {"timestamps": steps}

    def _draw(self, n: int, date, cond, rng, constrain: bool):
        C = self._sample_cond(n, date, cond, rng)
        X = self.model_.sample(n, cond=C, seed=int(rng.integers(2 ** 31)))
        V = self.scaler_.inverse_transform(self._unflatten(X))
        meta: Dict[str, Any] = {}
        ctx = self._context(date, n)
        cons = list(self.constraints or [])
        if cons and constrain:
            needs_ts = any(getattr(c, "envelope", None) is not None for c in cons)
            if ctx["timestamps"] is None and needs_ts:
                raise ValueError("a constraint needs timestamps; sample with `date=`")
            meta["violation_rates"] = violation_rates(V, cons, ctx, self.names_)
            V = apply_constraints(V, cons, ctx, self.names_)
        return V, ctx, meta

    def sample(self, n: int = 100, date=None, cond=None, seed: Optional[int] = None, constrain: bool = True,
               risk_alpha: Optional[float] = None, risk_loss="sum", oversample: int = 10,
               tail_share: float = 0.5) -> ScenarioSet:
        """Generate ``n`` scenarios in physical units.

        Parameters
        ----------
        n:
            Number of scenarios.
        date:
            Start date of the scenarios (needed for calendar conditions and
            for timestamp-aware constraints such as ``ClearSkyCap``).  A list
            of ``n`` dates gives per-scenario dates.
        cond:
            Custom condition values ``(k,)`` or ``(n, k)``.
        constrain:
            Apply physical constraints.
        risk_alpha:
            If set (e.g. ``0.95``), draw ``n * oversample`` candidates and
            return a stratified, probability-weighted set in which half of
            the scenarios (``tail_share``) describe the worst ``1 - alpha``
            of outcomes according to ``risk_loss``.  Yields much more
            accurate CVaR estimates for the same number of scenarios.
        """
        check_is_fitted(self)
        rng = make_rng(self.seed if seed is None else seed)
        if risk_alpha is None:
            V, ctx, meta = self._draw(n, date, cond, rng, constrain)
            probs = None
        else:
            if np.ndim(date) and len(np.atleast_1d(date)) > 1:
                raise ValueError("risk-aware sampling supports a single date")
            N = int(n * oversample)
            V, ctx, meta = self._draw(N, date, cond, rng, constrain)
            losses = scenario_loss(V, risk_loss)
            idx, probs = stratified_tail_sample(losses, n, risk_alpha, tail_share, rng)
            V = V[idx]
            meta.update(risk_alpha=risk_alpha, risk_loss=str(risk_loss), pool_size=N)
        ts = ctx["timestamps"][0] if ctx["timestamps"] is not None and (date is None or len(np.atleast_1d(date)) == 1) else None
        meta.update(model=type(self.model_).__name__, date=None if date is None else str(np.atleast_1d(date)[0]))
        return ScenarioSet(V, probs, ts, self.names_, self.unit_, meta)

    # --------------------------------------------------------- evaluation
    def evaluate(self, data=None, cond=None, n_per: int = 3, metrics=None, seed: int = 123) -> Dict[str, float]:
        """Compare generated with real profiles.

        With ``data=None`` the training profiles are used (in-sample); pass a
        hold-out series to evaluate out of sample.  For conditional models
        each real profile is matched with ``n_per`` samples drawn under its
        own conditions.
        """
        from ..scenarios.metrics import DEFAULT_METRICS, evaluate_scenarios

        check_is_fitted(self)
        if data is None:
            ref, C = self.reference_, self.C_
        else:
            ref, starts, cond = self._profiles(data, cond)
            C = self._cond_matrix(starts, cond, len(ref))
        n = len(ref) * n_per
        Cr = None if C is None else np.repeat(C, n_per, axis=0)
        X = self.model_.sample(n, cond=Cr, seed=seed)
        V = self.scaler_.inverse_transform(self._unflatten(X))
        cons = [c for c in (self.constraints or []) if not hasattr(c, "envelope")]
        if cons:
            V = apply_constraints(V, cons, {"timestamps": None}, self.names_)
        return evaluate_scenarios(ref, V, metrics or DEFAULT_METRICS)

    # ------------------------------------------------------------ drift
    def monitor(self, new_data, cond=None, alpha: float = 0.05, refit: bool = True,
                epochs: Optional[int] = None, replay: float = 1.0, method: str = "mmd") -> DriftResult:
        """Test new data for distribution drift and fine-tune if it occurred.

        The model is updated with ``partial_fit`` on the new profiles plus a
        replay sample of earlier data (``replay`` × the new data size) to avoid
        catastrophic forgetting.
        """
        check_is_fitted(self)
        prof, starts, cond = self._profiles(new_data, cond)
        Xn = self._flatten(self.scaler_.transform(prof))
        Cn = self._cond_matrix(starts, cond, len(prof))
        res = DriftDetector(self.X_, method=method, alpha=alpha, seed=self.seed).test(Xn)
        if res.drift and refit:
            rng = make_rng(self.seed)
            k = min(len(self.X_), int(replay * len(Xn)))
            old = rng.choice(len(self.X_), size=k, replace=False) if k else np.array([], dtype=int)
            Xu = np.concatenate([self.X_[old], Xn])
            Cu = None if Cn is None else np.concatenate([self.C_[old], Cn])
            self.model_.partial_fit(Xu, Cu, epochs=epochs)
            self.X_ = np.concatenate([self.X_, Xn])
            self.C_ = None if Cn is None else np.concatenate([self.C_, Cn])
            self.reference_ = np.concatenate([self.reference_, prof])
        return res

    # ---------------------------------------------------------- shortcuts
    @property
    def history(self):
        return getattr(self.model_, "history_", None)

    def plot_history(self, ax=None):
        from ..viz import plot_history

        return plot_history(self.history, ax=ax)

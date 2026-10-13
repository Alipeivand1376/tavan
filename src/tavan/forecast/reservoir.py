"""Next-Generation Reservoir Computing (NG-RC).

Gauthier et al. (*Nature Communications*, 2021) showed that a nonlinear
vector autoregression built from time-delayed inputs and their polynomial
products — with a linear readout fitted by ridge regression — matches
traditional echo-state reservoirs at a fraction of the cost and with no
random recurrent weights.  It is fast, deterministic and genuinely
different from the RNN family.
"""

from __future__ import annotations

import itertools

import numpy as np

from ..registry import register
from .base import BaseForecaster
from .linear import _ridge_fit, _ridge_predict


def _polynomial_features(lin: np.ndarray, order: int, interactions: bool) -> np.ndarray:
    """Monomials of ``lin`` up to ``order`` (optionally only pure powers)."""
    n, d = lin.shape
    feats = [np.ones((n, 1)), lin]
    if order >= 2:
        if interactions:
            idx = list(itertools.combinations_with_replacement(range(d), 2))
            feats.append(np.stack([lin[:, i] * lin[:, j] for i, j in idx], axis=1))
        else:
            feats.append(lin ** 2)
    if order >= 3:
        if interactions:
            idx = list(itertools.combinations_with_replacement(range(d), 3))
            feats.append(np.stack([lin[:, i] * lin[:, j] * lin[:, k] for i, j, k in idx], axis=1))
        else:
            feats.append(lin ** 3)
    return np.concatenate(feats, axis=1)


@register("forecaster", "ngrc", aliases=("ng_rc", "reservoir", "nvar"))
class NGRC(BaseForecaster):
    """Next-generation reservoir computer (nonlinear vector autoregression).

    Parameters
    ----------
    lookback:
        Number of past steps fed to the model.
    horizon:
        Forecast horizon.
    delays:
        Number of time-delay taps ``k``; the linear feature vector stacks the
        last ``k`` observations spaced by ``skip``.
    skip:
        Spacing between delay taps (``2`` keeps every other step).
    order:
        Maximum monomial order (2 or 3).
    interactions:
        Include cross terms between taps/channels (``False`` = pure powers,
        much smaller feature space).
    alpha:
        Ridge regularisation of the linear readout.
    strategy:
        ``"direct"`` (one readout per horizon step) or ``"recursive"``
        (predict one step and feed it back).

    Notes
    -----
    Recursive multi-step forecasting is only available for univariate series.
    """

    requires_windows = True
    _state_attrs = ("W_", "_min_len")

    def __init__(self, lookback: int = 48, horizon: int = 24, delays: int = 4, skip: int = 1,
                 order: int = 2, interactions: bool = False, alpha: float = 1e-2,
                 strategy: str = "direct") -> None:
        super().__init__(lookback=lookback, horizon=horizon)
        self.delays = delays
        self.skip = skip
        self.order = order
        self.interactions = interactions
        self.alpha = alpha
        self.strategy = strategy

    # ------------------------------------------------------------ features
    def _linear_part(self, window: np.ndarray) -> np.ndarray:
        """``window`` ``(n, L, C)`` -> delay-embedded linear vector ``(n, k*C)``."""
        taps = [window[:, -1 - i * self.skip, :] for i in range(self.delays)]
        return np.concatenate(taps, axis=1)

    def _features(self, window: np.ndarray) -> np.ndarray:
        return _polynomial_features(self._linear_part(window), self.order, self.interactions)

    def _fit_impl(self, y, X_exog):
        if self.strategy == "recursive" and self.n_channels_ > 1:
            raise ValueError("recursive strategy supports only univariate series")
        min_len = (self.delays - 1) * self.skip + 1
        self._min_len = min_len
        if self.strategy == "recursive":
            Xw, Yw, _ = self._make_windows(y)
            F = self._features(Xw)
            self.W_ = _ridge_fit(F, Yw[:, 0, :], self.alpha)
        else:
            Xw, Yw, _ = self._make_windows(y)
            F = self._features(Xw)
            self.W_ = [_ridge_fit(F, Yw[:, h, :], self.alpha) for h in range(self.horizon)]

    def _forecast_impl(self, context, X_exog, horizon):
        L = max(self.lookback, self._min_len)
        ctx = context[-L:]
        if len(ctx) < L:
            ctx = np.pad(ctx, ((L - len(ctx), 0), (0, 0)), mode="edge")
        if self.strategy == "recursive":
            buf = ctx.copy()
            out = []
            for _ in range(horizon):
                F = self._features(buf[None, -self.lookback:])
                nxt = _ridge_predict(self.W_, F)[0]
                out.append(nxt)
                buf = np.concatenate([buf, nxt[None, :]], axis=0)
            return self._wrap(np.array(out))
        F = self._features(ctx[None, -self.lookback:])
        preds = [_ridge_predict(self.W_[h], F)[0] for h in range(min(horizon, self.horizon))]
        out = np.array(preds)
        if horizon > self.horizon:
            out = np.pad(out, ((0, horizon - self.horizon), (0, 0)), mode="edge")
        return self._wrap(out)

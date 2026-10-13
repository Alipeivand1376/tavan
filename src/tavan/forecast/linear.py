"""Linear forecasters: DLinear and NLinear (Zeng et al., AAAI 2023).

These are strikingly strong baselines — a single linear layer over the
look-back window, applied per channel.  DLinear first splits the series into
a moving-average trend and a remainder and models each with its own linear
map.  They are fitted here by ridge regression in closed form (no gradient
descent), so training takes milliseconds.
"""

from __future__ import annotations

import numpy as np

from ..registry import register
from .base import BaseForecaster


def _ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Solve ``min ||X W - Y||^2 + alpha ||W||^2`` with an intercept column."""
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    d = Xb.shape[1]
    A = Xb.T @ Xb
    reg = alpha * np.eye(d)
    reg[-1, -1] = 0.0  # do not penalise the intercept
    return np.linalg.solve(A + reg, Xb.T @ Y)


def _ridge_predict(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1) @ W


def moving_average(x: np.ndarray, kernel: int) -> np.ndarray:
    """Centred moving average along axis 1 with edge padding (per channel)."""
    if kernel <= 1:
        return x
    pad = kernel // 2
    xp = np.pad(x, ((0, 0), (pad, kernel - 1 - pad), (0, 0)), mode="edge")
    csum = np.cumsum(xp, axis=1)
    csum = np.concatenate([np.zeros((x.shape[0], 1, x.shape[2])), csum], axis=1)
    out = (csum[:, kernel:] - csum[:, :-kernel]) / kernel
    return out[:, : x.shape[1]]


class _LinearBase(BaseForecaster):
    requires_windows = True

    def __init__(self, lookback: int = 48, horizon: int = 24, alpha: float = 1.0,
                 individual: bool = True) -> None:
        super().__init__(lookback=lookback, horizon=horizon)
        self.alpha = alpha
        self.individual = individual

    def _prep(self, Xw):  # pragma: no cover - abstract
        raise NotImplementedError

    def _fit_impl(self, y, X_exog):
        Xw, Yw, _ = self._make_windows(y)  # (n, L, C), (n, H, C)
        C = self.n_channels_
        self.W_ = []
        for c in range(C):
            xc = Xw[:, :, c]
            feats = self._prep_channel(xc)
            self.W_.append([_ridge_fit(f, Yw[:, :, c], self.alpha) for f in feats])

    def _forecast_impl(self, context, X_exog, horizon):
        L = self.lookback
        ctx = context[-L:]
        if len(ctx) < L:
            ctx = np.pad(ctx, ((L - len(ctx), 0), (0, 0)), mode="edge")
        preds = []
        for c in range(self.n_channels_):
            feats = self._prep_channel(ctx[:, c][None, :])
            p = sum(_ridge_predict(W, f)[0] for W, f in zip(self.W_[c], feats))
            preds.append(p[:horizon] if horizon <= self.horizon else np.pad(p, (0, horizon - self.horizon), mode="edge"))
        return self._wrap(np.column_stack(preds))


@register("forecaster", "nlinear")
class NLinear(_LinearBase):
    _state_attrs = ("W_",)
    """NLinear: subtract the last value, apply one linear map, add it back.

    Robust to distribution shift between training and test windows.
    """

    def _prep_channel(self, xc):
        last = xc[:, -1:]
        return [(xc - last)]  # intercept handled in ridge; last added back at predict

    def _fit_impl(self, y, X_exog):
        Xw, Yw, _ = self._make_windows(y)
        self.W_ = []
        for c in range(self.n_channels_):
            xc = Xw[:, :, c]
            last = xc[:, -1:]
            self.W_.append(_ridge_fit(xc - last, Yw[:, :, c] - last, self.alpha))

    def _forecast_impl(self, context, X_exog, horizon):
        L = self.lookback
        ctx = context[-L:]
        if len(ctx) < L:
            ctx = np.pad(ctx, ((L - len(ctx), 0), (0, 0)), mode="edge")
        preds = []
        for c in range(self.n_channels_):
            xc = ctx[:, c][None, :]
            last = xc[:, -1:]
            p = (_ridge_predict(self.W_[c], xc - last) + last)[0]
            preds.append(np.pad(p, (0, max(0, horizon - self.horizon)), mode="edge")[:horizon])
        return self._wrap(np.column_stack(preds))


@register("forecaster", "dlinear")
class DLinear(_LinearBase):
    _state_attrs = ("W_", "kernel")
    """DLinear: decompose into trend + seasonal remainder, one linear map each.

    Parameters
    ----------
    lookback, horizon:
        Window lengths.
    kernel:
        Moving-average kernel for the trend (e.g. 25 for hourly daily data).
    alpha:
        Ridge regularisation.
    """

    def __init__(self, lookback: int = 48, horizon: int = 24, kernel: int = 25, alpha: float = 1.0,
                 individual: bool = True) -> None:
        super().__init__(lookback=lookback, horizon=horizon, alpha=alpha, individual=individual)
        self.kernel = kernel

    def _decompose_channel(self, xc):
        x3 = xc[:, :, None]
        trend = moving_average(x3, self.kernel)[:, :, 0]
        return trend, xc - trend

    def _fit_impl(self, y, X_exog):
        Xw, Yw, _ = self._make_windows(y)
        # one joint ridge on [trend | seasonal] per channel (optimal linear combination)
        self.W_ = []
        for c in range(self.n_channels_):
            trend, seas = self._decompose_channel(Xw[:, :, c])
            feat = np.concatenate([trend, seas], axis=1)
            self.W_.append(_ridge_fit(feat, Yw[:, :, c], self.alpha))

    def _forecast_impl(self, context, X_exog, horizon):
        L = self.lookback
        ctx = context[-L:]
        if len(ctx) < L:
            ctx = np.pad(ctx, ((L - len(ctx), 0), (0, 0)), mode="edge")
        preds = []
        for c in range(self.n_channels_):
            trend, seas = self._decompose_channel(ctx[:, c][None, :])
            feat = np.concatenate([trend, seas], axis=1)
            p = _ridge_predict(self.W_[c], feat)[0]
            preds.append(np.pad(p, (0, max(0, horizon - self.horizon)), mode="edge")[:horizon])
        return self._wrap(np.column_stack(preds))

"""Modern neural forecasters (no RNNs): N-HiTS, TiDE, TSMixer, PatchTST.

All are multi-step, direct forecasters trained with the shared NumPy
autograd engine.  They accept exogenous/future covariates and can be fitted
with a quantile (pinball) loss for native probabilistic forecasts.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..config import TrainConfig
from ..nn import (
    MLP,
    Linear,
    Module,
    ResidualBlock,
    Tensor,
    TransformerEncoderLayer,
    concat,
    huber,
    mse,
    no_grad,
    pinball,
    sinusoidal_embedding,
    train,
)
from ..registry import register
from .base import BaseForecaster


class _NeuralForecaster(BaseForecaster):
    """Shared training / windowing logic for neural forecasters."""

    requires_windows = True
    supports_quantiles = True
    supports_exog = True
    _module_attrs = ("net_",)

    def __init__(self, lookback: int = 48, horizon: int = 24, quantiles: Optional[Sequence[float]] = None,
                 loss: str = "huber", epochs: int = 150, lr: float = 1e-3, batch_size: int = 64,
                 dropout: float = 0.0, max_train_windows: Optional[int] = 8000, seed: int = 0,
                 verbose: int = 0, train_config=None) -> None:
        super().__init__(lookback=lookback, horizon=horizon)
        self.quantiles = tuple(quantiles) if quantiles else None
        self.loss = loss
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.max_train_windows = max_train_windows
        self.seed = seed
        self.verbose = verbose
        self.train_config = train_config

    @property
    def n_outputs(self) -> int:
        return len(self.quantiles) if self.quantiles else 1

    def _build(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def _forward(self, net, xb, fb):  # pragma: no cover - abstract
        raise NotImplementedError

    def _standardize(self, Xw, Yw):
        self.mu_ = Xw.mean(axis=(0, 1), keepdims=True)
        self.sd_ = Xw.std(axis=(0, 1), keepdims=True) + 1e-6
        return (Xw - self.mu_) / self.sd_, (Yw - self.mu_) / self.sd_

    def _loss_fn(self, net, xb, yb, fb, rng=None, epoch=0):
        pred = self._forward(net, xb, fb)
        if self.quantiles:
            return pinball(pred, yb, self.quantiles)
        base = {"mse": mse, "huber": huber}[self.loss]
        return base(pred, yb)

    def _fit_impl(self, y, X_exog):
        Xw, Yw, Fw = self._make_windows(y, X_exog)
        cap = self.max_train_windows
        if cap and len(Xw) > cap:
            # even stride keeps the series' chronological coverage
            idx = np.linspace(0, len(Xw) - 1, cap).round().astype(int)
            Xw, Yw = Xw[idx], Yw[idx]
            Fw = None if Fw is None else Fw[idx]
        Xs, Ys = self._standardize(Xw, Yw)
        if Fw is not None:
            self.fmu_ = Fw.mean((0, 1), keepdims=True)
            self.fsd_ = Fw.std((0, 1), keepdims=True) + 1e-6
            Fw = (Fw - self.fmu_) / self.fsd_
        self._build()
        cfg = TrainConfig.from_any(self.train_config, epochs=self.epochs, lr=self.lr,
                                   batch_size=self.batch_size, seed=self.seed, verbose=self.verbose)
        arrays = [Xs, Ys] + ([Fw] if Fw is not None else [None])
        self.history_ = train(self.net_, self._loss_fn, arrays, cfg)

    def _forecast_impl(self, context, X_exog, horizon):
        L = self.lookback
        ctx = context[-L:]
        if len(ctx) < L:
            ctx = np.pad(ctx, ((L - len(ctx), 0), (0, 0)), mode="edge")
        xs = (ctx[None] - self.mu_) / self.sd_
        fb = None
        if X_exog is not None:
            fb = np.asarray(X_exog, dtype=float)
            fb = fb[None, :self.horizon] if fb.ndim == 2 else fb.reshape(1, self.horizon, -1)
            fb = (fb - self.fmu_) / self.fsd_
        net = self.net_
        net.eval()
        with no_grad():
            pred = self._forward(net, xs, fb).data[0]  # (H, C) or (H, C, Q)
        if self.quantiles:
            pred = pred.reshape(self.horizon, self.n_channels_, self.n_outputs)
            qs = {}
            for i, q in enumerate(self.quantiles):
                qs[q] = pred[:, :, i] * self.sd_[0] + self.mu_[0]
            mean = qs[min(self.quantiles, key=lambda q: abs(q - 0.5))]
            out = self._trim(mean, horizon)
            qs = {k: self._trim(v, horizon) for k, v in qs.items()}
            return self._wrap(out, qs)
        mean = pred.reshape(self.horizon, self.n_channels_) * self.sd_[0] + self.mu_[0]
        return self._wrap(self._trim(mean, horizon))

    def _trim(self, arr, horizon):
        if horizon <= self.horizon:
            return arr[:horizon]
        return np.pad(arr, ((0, horizon - self.horizon), (0, 0)), mode="edge")

    # ---- neural state persistence -------------------------------------
    def _get_state(self):
        st = {a: getattr(self, a) for a in ("mu_", "sd_", "n_channels_", "n_exog_", "is_fitted_", "_was_1d")}
        for a in ("fmu_", "fsd_"):
            if hasattr(self, a):
                st[a] = getattr(self, a)
        st["net_"] = self.net_
        return st

    def _set_state(self, state):
        net = state.pop("net_")
        for k, v in state.items():
            setattr(self, k, v)
        self._build()
        self.net_.load_state_dict(net)
        self.net_.eval()

    @property
    def history(self):
        return getattr(self, "history_", None)


# ===================================================================== N-HiTS
class _NHiTSBlock(Module):
    def __init__(self, Lin, H, C, nout, hidden, pool, theta, activation, dropout, rng):
        self.pool = pool
        self.C = C
        self.nout = nout
        pooled = (Lin + pool - 1) // pool
        self.mlp = MLP([pooled * C, *hidden, (theta + H) * C * nout], activation=activation,
                       dropout=dropout, rng=rng)
        self.theta = theta
        self.H = H
        self.Lin = Lin

    def forward(self, x):  # x: (n, L, C)
        n = x.shape[0]
        xp = x
        if self.pool > 1:
            Lp = (self.Lin // self.pool) * self.pool
            xp = x[:, self.Lin - Lp:].reshape(n, Lp // self.pool, self.pool, self.C).mean(axis=2)
        h = self.mlp(xp.reshape(n, -1))
        h = h.reshape(n, self.theta + self.H, self.C, self.nout)
        back = h[:, : self.theta]
        fore = h[:, self.theta:]
        return back, fore


class _NHiTS(Module):
    def __init__(self, L, H, C, nout, pools, hidden, activation, dropout, rng):
        self.blocks = []
        self.L, self.H, self.C, self.nout = L, H, C, nout
        for p in pools:
            theta = max(1, L // p)
            self.blocks.append(_NHiTSBlock(L, H, C, nout, hidden, p, theta, activation, dropout, rng))

    def forward(self, x):
        # multi-rate residual stacking with linear interpolation of backcast
        residual = x
        forecast = None
        for b in self.blocks:
            back, fore = b(x if b is self.blocks[0] else residual)
            bc = self._interp(back[..., 0], self.L)
            residual = residual - bc
            forecast = fore if forecast is None else forecast + fore
        return forecast  # (n, H, C, nout)

    @staticmethod
    def _interp(y, out_len):
        # y: (n, theta, C) -> (n, out_len, C) via linear interpolation over time
        n, theta, C = y.shape
        if theta == out_len:
            return y
        src = np.linspace(0, 1, theta)
        tgt = np.linspace(0, 1, out_len)
        idx = np.clip(np.searchsorted(src, tgt) - 1, 0, theta - 2)
        w = ((tgt - src[idx]) / (src[idx + 1] - src[idx]))[None, :, None]
        left = y[:, idx, :]
        right = y[:, idx + 1, :]
        return left * (1 - Tensor(w)) + right * Tensor(w)


@register("forecaster", "nhits", aliases=("n_hits", "nbeats"))
class NHiTS(_NeuralForecaster):
    """N-HiTS (Challu et al., AAAI 2023): multi-rate hierarchical MLP stacks.

    Pooling at several rates lets early blocks model low-frequency trend and
    later blocks the high-frequency detail; backcasts are subtracted
    residually.  Strong and fast for long-horizon forecasting.
    """

    def __init__(self, lookback: int = 96, horizon: int = 24, pools: Sequence[int] = (4, 2, 1),
                 hidden: Sequence[int] = (256, 256), activation: str = "relu", **kw) -> None:
        super().__init__(lookback=lookback, horizon=horizon, **kw)
        self.pools = tuple(pools)
        self.hidden = tuple(hidden)
        self.activation = activation

    def _build(self):
        self.net_ = _NHiTS(self.lookback, self.horizon, self.n_channels_, self.n_outputs,
                           self.pools, self.hidden, self.activation, self.dropout,
                           np.random.default_rng(self.seed))

    def _forward(self, net, xb, fb):
        out = net(Tensor(xb))
        return out if self.quantiles else out[..., 0]


# ====================================================================== TiDE
class _TiDE(Module):
    def __init__(self, L, H, C, nout, n_exog, hidden, n_blocks, decoder_dim, dropout, rng):
        self.L, self.H, self.C, self.nout, self.n_exog = L, H, C, nout, n_exog
        din = L * C + (H * n_exog if n_exog else 0)
        self.enc = [ResidualBlock(din if i == 0 else hidden, hidden, hidden, dropout, rng=rng) for i in range(n_blocks)]
        self.dec = [ResidualBlock(hidden if i == 0 else decoder_dim, decoder_dim, decoder_dim, dropout, rng=rng)
                    for i in range(n_blocks)]
        self.proj = Linear(decoder_dim, H * C * nout, rng=rng)
        self.skip = Linear(L, H, rng=rng)

    def forward(self, x, f=None):
        n = x.shape[0]
        parts = [x.reshape(n, -1)]
        if self.n_exog and f is not None:
            parts.append(f.reshape(n, -1))
        h = concat(parts, axis=1) if len(parts) > 1 else parts[0]
        for blk in self.enc:
            h = blk(h)
        for blk in self.dec:
            h = blk(h)
        out = self.proj(h).reshape(n, self.H, self.C, self.nout)
        skip = self.skip(x.swapaxes(1, 2)).swapaxes(1, 2)  # (n, H, C)
        return out + skip.reshape(n, self.H, self.C, 1)


@register("forecaster", "tide")
class TiDE(_NeuralForecaster):
    """TiDE (Das et al., 2023): a dense encoder–decoder with residual blocks.

    A pure-MLP long-horizon model that handles exogenous covariates well and
    trains quickly; includes a linear skip from inputs to outputs.
    """

    def __init__(self, lookback: int = 96, horizon: int = 24, hidden: int = 256, n_blocks: int = 2,
                 decoder_dim: int = 128, **kw) -> None:
        super().__init__(lookback=lookback, horizon=horizon, **kw)
        self.hidden = hidden
        self.n_blocks = n_blocks
        self.decoder_dim = decoder_dim

    def _build(self):
        self.net_ = _TiDE(self.lookback, self.horizon, self.n_channels_, self.n_outputs, self.n_exog_,
                          self.hidden, self.n_blocks, self.decoder_dim, self.dropout,
                          np.random.default_rng(self.seed))

    def _forward(self, net, xb, fb):
        out = net(Tensor(xb), None if fb is None else Tensor(fb))
        return out if self.quantiles else out[..., 0]


# ==================================================================== TSMixer
class _MixerLayer(Module):
    def __init__(self, L, C, token_dim, channel_dim, dropout, rng):
        from ..nn import LayerNorm

        self.norm1 = LayerNorm(C)
        self.time = MLP([L, token_dim, L], activation="gelu", dropout=dropout, rng=rng)
        self.norm2 = LayerNorm(C)
        self.chan = MLP([C, channel_dim, C], activation="gelu", dropout=dropout, rng=rng)

    def forward(self, x):  # (n, L, C)
        h = self.norm1(x)
        h = h.swapaxes(1, 2)          # (n, C, L)
        h = self.time(h).swapaxes(1, 2)
        x = x + h
        return x + self.chan(self.norm2(x))


class _TSMixer(Module):
    def __init__(self, L, H, C, nout, n_layers, token_dim, channel_dim, dropout, rng):
        self.layers = [_MixerLayer(L, C, token_dim, channel_dim, dropout, rng) for _ in range(n_layers)]
        self.head = Linear(L, H * nout, rng=rng)
        self.H, self.C, self.nout = H, C, nout

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        h = self.head(x.swapaxes(1, 2))  # (n, C, H*nout)
        n = x.shape[0]
        return h.reshape(n, self.C, self.H, self.nout).transpose(0, 2, 1, 3)


@register("forecaster", "tsmixer", aliases=("mixer",))
class TSMixer(_NeuralForecaster):
    """TSMixer (Chen et al., 2023): all-MLP mixing across time and channels.

    Alternating time-mixing and channel-mixing MLPs capture cross-series
    dependencies — useful for jointly forecasting correlated feeders or
    multiple sites.
    """

    def __init__(self, lookback: int = 96, horizon: int = 24, n_layers: int = 3, token_dim: int = 128,
                 channel_dim: int = 64, **kw) -> None:
        super().__init__(lookback=lookback, horizon=horizon, **kw)
        self.n_layers = n_layers
        self.token_dim = token_dim
        self.channel_dim = channel_dim

    def _build(self):
        self.net_ = _TSMixer(self.lookback, self.horizon, self.n_channels_, self.n_outputs, self.n_layers,
                             self.token_dim, self.channel_dim, self.dropout, np.random.default_rng(self.seed))

    def _forward(self, net, xb, fb):
        out = net(Tensor(xb))
        return out if self.quantiles else out[..., 0]


# ==================================================================== PatchTST
class _PatchTST(Module):
    def __init__(self, L, H, C, nout, patch, stride, d_model, n_heads, n_layers, d_ff, dropout, rng):
        self.patch, self.stride, self.C = patch, stride, C
        self.n_patches = (L - patch) // stride + 1
        self.embed = Linear(patch, d_model, rng=rng)
        self.pos = sinusoidal_embedding(np.arange(self.n_patches), d_model)
        self.layers = [TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, rng=rng) for _ in range(n_layers)]
        self.head = Linear(self.n_patches * d_model, H * nout, rng=rng)
        self.H, self.nout, self.d_model = H, nout, d_model

    def forward(self, x):  # (n, L, C) — channel independent
        n, L, C = x.shape
        starts = [i * self.stride for i in range(self.n_patches)]
        patches = concat([x[:, s:s + self.patch, :].swapaxes(1, 2).reshape(n * C, 1, self.patch)
                          for s in starts], axis=1)  # (n*C, n_patches, patch)
        h = self.embed(patches) + Tensor(self.pos)[None]
        for layer in self.layers:
            h = layer(h)
        out = self.head(h.reshape(n * C, -1)).reshape(n, C, self.H, self.nout)
        return out.transpose(0, 2, 1, 3)


@register("forecaster", "patchtst", aliases=("patch_tst",))
class PatchTST(_NeuralForecaster):
    """PatchTST (Nie et al., ICLR 2023): patch the series, apply a Transformer.

    Channel-independent patching gives a strong attention-based forecaster.
    Heavier than the MLP models, so best reserved for longer series.
    """

    def __init__(self, lookback: int = 96, horizon: int = 24, patch: int = 16, stride: int = 8,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2, d_ff: int = 128, **kw) -> None:
        super().__init__(lookback=lookback, horizon=horizon, **kw)
        self.patch = patch
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff

    def _build(self):
        self.net_ = _PatchTST(self.lookback, self.horizon, self.n_channels_, self.n_outputs, self.patch,
                              self.stride, self.d_model, self.n_heads, self.n_layers, self.d_ff, self.dropout,
                              np.random.default_rng(self.seed))

    def _forward(self, net, xb, fb):
        out = net(Tensor(xb))
        return out if self.quantiles else out[..., 0]

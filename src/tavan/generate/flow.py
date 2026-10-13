"""RealNVP normalising flow with exact likelihood."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from ..nn import MLP, Module, Parameter, Tensor, concat, no_grad, train
from ..registry import register
from .base import NeuralGenerator


class _Coupling(Module):
    def __init__(self, d, k, hidden, perm, rng):
        self.perm = perm
        self.inv = np.argsort(perm)
        self.da = d // 2
        self.db = d - self.da
        self.net = MLP([self.da + k, *hidden, 2 * self.db], activation="gelu", rng=rng)
        # start as identity map
        last = self.net.linears[-1]
        last.weight.data[:] = 0.0
        self.scale = Parameter(np.ones(self.db))

    def _st(self, xa: Tensor, c):
        h = self.net(xa if c is None else concat([xa, c], axis=1))
        s = h[:, : self.db].tanh() * self.scale
        t = h[:, self.db:]
        return s, t

    def forward(self, x: Tensor, c=None):
        """x -> z, log|det J|."""
        xp = x[:, self.perm]
        xa, xb = xp[:, : self.da], xp[:, self.da:]
        s, t = self._st(xa, c)
        yb = xb * s.exp() + t
        return concat([xa, yb], axis=1), s.sum(axis=1)

    def inverse(self, z: Tensor, c=None):
        za, zb = z[:, : self.da], z[:, self.da:]
        s, t = self._st(za, c)
        xb = (zb - t) * (-s).exp()
        return concat([za, xb], axis=1)[:, self.inv]


class _Flow(Module):
    def __init__(self, d, k, n_layers, hidden, rng):
        self.layers = [_Coupling(d, k, hidden, rng.permutation(d), rng) for _ in range(n_layers)]

    def forward(self, x, c=None):
        logdet = 0.0
        for layer in self.layers:
            x, ld = layer(x, c)
            logdet = ld + logdet
        return x, logdet

    def inverse(self, z, c=None):
        for layer in reversed(self.layers):
            z = layer.inverse(z, c)
        return z


@register("generator", "flow", aliases=("realnvp", "normalizing_flow"))
class RealNVP(NeuralGenerator):
    """Real-valued non-volume-preserving flow (Dinh et al., 2017).

    A stack of affine coupling layers with random feature permutations maps
    the data to a standard normal.  Because the transform is invertible, the
    model provides **exact log-likelihoods** (:meth:`score_samples`), which
    makes it useful for anomaly detection (e.g. flagging attacked or faulty
    measurements) in addition to scenario generation.

    Parameters
    ----------
    n_layers:
        Number of coupling layers.
    hidden:
        Hidden sizes of each coupling network.
    temperature:
        Scale of the base noise when sampling (<1 gives more typical samples).
    """

    supports_likelihood = True
    _module_attrs = ("net_",)
    _state_attrs = ()

    def __init__(self, n_layers: int = 6, hidden: Sequence[int] = (128, 128), temperature: float = 1.0,
                 epochs: Optional[int] = 300, lr: Optional[float] = 1e-3, batch_size: Optional[int] = 64,
                 seed: Optional[int] = 0, verbose: Optional[int] = 0, train=None) -> None:
        self.n_layers = n_layers
        self.hidden = hidden
        self.temperature = temperature
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose
        self.train = train

    def _build(self):
        if self.n_features_ < 2:
            raise ValueError("RealNVP needs at least 2 features")
        self.net_ = _Flow(self.n_features_, self.n_cond_, self.n_layers, tuple(self.hidden),
                          np.random.default_rng(self.seed))

    def _nll(self, net, xb, cb, rng=None, epoch=0):
        z, logdet = net(Tensor(xb), None if cb is None else Tensor(cb))
        d = xb.shape[1]
        logp = (z * z).sum(axis=1) * -0.5 - 0.5 * d * math.log(2 * math.pi)
        return -(logp + logdet).mean() * (1.0 / d)

    def _fit(self, X, cond):
        self._build()
        self.history_ = train(self.net_, self._nll, [X, cond], self._train_config())

    def partial_fit(self, X, cond=None, epochs: Optional[int] = None):
        if not getattr(self, "is_fitted_", False):
            return self.fit(X, cond)
        cond = self._check_cond(cond, len(X))
        cfg = self._train_config(epochs=epochs or max(20, (self.epochs or 100) // 5))
        self.history_ = train(self.net_, self._nll, [np.asarray(X, float), cond], cfg)
        return self

    def _sample(self, n, cond, rng):
        z = rng.normal(size=(n, self.n_features_)) * self.temperature
        with no_grad():
            return self.net_.inverse(Tensor(z), None if cond is None else Tensor(cond)).data

    def score_samples(self, X, cond=None) -> np.ndarray:
        """Exact log-density ``log p(x | c)`` per sample."""
        X = np.asarray(X, dtype=float)
        with no_grad():
            z, logdet = self.net_(Tensor(X), None if cond is None else Tensor(np.atleast_2d(cond)))
        d = X.shape[1]
        logdet = logdet.data if isinstance(logdet, Tensor) else logdet
        return -0.5 * (z.data ** 2).sum(1) - 0.5 * d * math.log(2 * math.pi) + logdet

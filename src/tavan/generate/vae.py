"""(Conditional) variational auto-encoder."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..nn import MLP, Parameter, Tensor, concat, no_grad, train
from ..nn.layers import Module
from ..registry import register
from .base import NeuralGenerator, _nn_cond


class _VAENet(Module):
    def __init__(self, d, k, latent, hidden, activation, rng):
        self.enc = MLP([d + k, *hidden, 2 * latent], activation=activation, rng=rng)
        self.dec = MLP([latent + k, *reversed(hidden), d], activation=activation, rng=rng)
        self.log_sigma = Parameter(np.full(d, np.log(0.3)))
        self.latent = latent


@register("generator", "vae", aliases=("cvae",))
class VAE(NeuralGenerator):
    """Conditional variational auto-encoder (Kingma & Welling, 2014).

    The decoder models ``p(x | z, c)`` as a Gaussian with a learned
    per-feature scale, so samples include realistic residual noise.  KL
    annealing (``kl_warmup``) avoids posterior collapse.

    Parameters
    ----------
    latent_dim:
        Size of the latent space.
    hidden:
        Hidden layer sizes of encoder (decoder mirrors them).
    beta:
        Weight of the KL term (``beta-VAE``).
    kl_warmup:
        Epochs over which the KL weight ramps up from 0 to ``beta``.
    sample_noise:
        Add decoder noise when sampling (``False`` returns decoder means).
    epochs, lr, batch_size, seed, verbose:
        Shortcuts for the most common :class:`~tavan.config.TrainConfig` fields.
    train:
        Full :class:`~tavan.config.TrainConfig` (or dict) for other settings.
    """

    _module_attrs = ("net_",)
    _state_attrs = ()

    def __init__(self, latent_dim: int = 8, hidden: Sequence[int] = (128, 128), beta: float = 1.0,
                 kl_warmup: int = 50, activation: str = "gelu", sample_noise: bool = True,
                 epochs: Optional[int] = 300, lr: Optional[float] = 2e-3, batch_size: Optional[int] = 64,
                 seed: Optional[int] = 0, verbose: Optional[int] = 0, train=None) -> None:
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.beta = beta
        self.kl_warmup = kl_warmup
        self.activation = activation
        self.sample_noise = sample_noise
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose
        self.train = train

    def _build(self):
        self.net_ = _VAENet(self.n_features_, self.n_cond_, self.latent_dim, tuple(self.hidden),
                            self.activation, np.random.default_rng(self.seed))

    def _loss(self, net, xb, cb, rng=None, epoch=0):
        L = net.latent
        h = net.enc(Tensor(_nn_cond(xb, cb)))
        mu, logv = h[:, :L], h[:, L:].clip(-10, 10)
        eps = rng.normal(size=mu.shape)
        z = mu + (logv * 0.5).exp() * eps
        dec_in = z if cb is None else concat([z, Tensor(cb)], axis=1)
        xr = net.dec(dec_in)
        ls = net.log_sigma.clip(-5, 3)
        diff = (Tensor(xb) - xr) / ls.exp()
        rec = ((diff * diff) * 0.5 + ls).sum(axis=1).mean()
        kl = ((mu * mu + logv.exp() - logv - 1.0) * 0.5).sum(axis=1).mean()
        # annealed KL weight while training; validation always uses the true objective
        if self.kl_warmup and net.training:
            w = self.beta * min(1.0, (epoch + 1) / max(1, self.kl_warmup))
        else:
            w = self.beta
        return (rec + kl * w) * (1.0 / xb.shape[1])

    def _fit(self, X, cond):
        self._build()
        self.history_ = train(self.net_, self._loss, [X, cond], self._train_config())

    def partial_fit(self, X, cond=None, epochs: Optional[int] = None):
        if not getattr(self, "is_fitted_", False):
            return self.fit(X, cond)
        X = np.asarray(X, dtype=float)
        cond = self._check_cond(cond, len(X))
        cfg = self._train_config(epochs=epochs or max(10, (self.epochs or 100) // 5), warmup=0)
        self.kl_warmup, kw = 0, self.kl_warmup
        try:
            self.history_ = train(self.net_, self._loss, [X, cond], cfg)
        finally:
            self.kl_warmup = kw
        return self

    def _sample(self, n, cond, rng):
        net = self.net_
        with no_grad():
            z = Tensor(rng.normal(size=(n, net.latent)))
            dec_in = z if cond is None else concat([z, Tensor(cond)], axis=1)
            x = net.dec(dec_in).data
        if self.sample_noise:
            x = x + rng.normal(size=x.shape) * np.exp(np.clip(net.log_sigma.data, -5, 3))
        return x

    def encode(self, X, cond=None) -> np.ndarray:
        """Posterior means of the latent codes (useful for clustering/visualisation)."""
        X = np.asarray(X, dtype=float)
        with no_grad():
            h = self.net_.enc(Tensor(_nn_cond(X, None if cond is None else np.asarray(cond, float)))).data
        return h[:, : self.latent_dim]

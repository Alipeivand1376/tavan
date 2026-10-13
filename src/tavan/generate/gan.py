"""Conditional generative adversarial network."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..nn import MLP, Module, Tensor, bce_with_logits, concat, no_grad
from ..nn.optim import Adam, clip_grad_norm
from ..nn.trainer import History
from ..registry import register
from .base import NeuralGenerator


class _GAN(Module):
    def __init__(self, d, k, noise_dim, g_hidden, d_hidden, rng):
        self.G = MLP([noise_dim + k, *g_hidden, d], activation="gelu", rng=rng)
        self.D = MLP([d + k, *d_hidden, 1], activation="leaky_relu", dropout=0.1, rng=rng)
        self.noise_dim = noise_dim


@register("generator", "gan", aliases=("cgan",))
class CGAN(NeuralGenerator):
    """Conditional GAN (Mirza & Osindero, 2014) with stabilising tricks.

    * non-saturating loss with one-sided label smoothing;
    * instance noise on discriminator inputs that decays during training;
    * an optional *moment-matching* term on the generator (mean, standard
      deviation and lag-1 correlation of each feature) that markedly
      improves stability on the small datasets common in power systems.

    Parameters
    ----------
    noise_dim:
        Size of the generator input noise.
    g_hidden, d_hidden:
        Hidden sizes of generator and discriminator.
    moment_weight:
        Weight of the moment-matching loss (0 = pure adversarial).
    label_smoothing:
        Target for real samples (e.g. 0.9).
    instance_noise:
        Initial std of Gaussian noise added to discriminator inputs.
    n_critic:
        Discriminator updates per generator update.
    """

    _module_attrs = ("net_",)
    _state_attrs = ()

    def __init__(self, noise_dim: int = 32, g_hidden: Sequence[int] = (128, 128),
                 d_hidden: Sequence[int] = (128, 128), moment_weight: float = 1.0,
                 label_smoothing: float = 0.9, instance_noise: float = 0.1, n_critic: int = 1,
                 epochs: Optional[int] = 400, lr: Optional[float] = 2e-4, batch_size: Optional[int] = 64,
                 seed: Optional[int] = 0, verbose: Optional[int] = 0, train=None) -> None:
        self.noise_dim = noise_dim
        self.g_hidden = g_hidden
        self.d_hidden = d_hidden
        self.moment_weight = moment_weight
        self.label_smoothing = label_smoothing
        self.instance_noise = instance_noise
        self.n_critic = n_critic
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose
        self.train = train

    def _build(self):
        self.net_ = _GAN(self.n_features_, self.n_cond_, self.noise_dim, tuple(self.g_hidden),
                         tuple(self.d_hidden), np.random.default_rng(self.seed))

    def _gen(self, z, c):
        return self.net_.G(z if c is None else concat([z, c], axis=1))

    def _disc(self, x, c):
        return self.net_.D(x if c is None else concat([x, c], axis=1))

    @staticmethod
    def _moments(x: Tensor):
        mu = x.mean(axis=0)
        xc = x - mu
        sd = ((xc * xc).mean(axis=0) + 1e-6).sqrt()
        if x.shape[1] > 1:
            lag = (xc[:, 1:] * xc[:, :-1]).mean(axis=0) / (sd[1:] * sd[:-1])
        else:
            lag = None
        return mu, sd, lag

    def _run(self, X, cond, epochs):
        cfg = self._train_config(epochs=epochs)
        rng = np.random.default_rng(cfg.seed)
        net = self.net_
        optG = Adam(net.G.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
        optD = Adam(net.D.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
        hist = getattr(self, "history_", None) or History()
        n = len(X)
        bs = max(2, min(cfg.batch_size, n))
        real_t = self.label_smoothing
        for epoch in range(cfg.epochs):
            noise_sd = self.instance_noise * max(0.0, 1 - epoch / max(1, cfg.epochs))
            perm = rng.permutation(n)
            g_tot = d_tot = 0.0
            nb = 0
            for s in range(0, n - 1, bs):
                bi = perm[s:s + bs]
                if len(bi) < 2:
                    continue
                xb = X[bi]
                cb = None if cond is None else Tensor(cond[bi])
                m = len(bi)
                net.train()
                for _ in range(self.n_critic):
                    with no_grad():
                        fake = self._gen(Tensor(rng.normal(size=(m, net.noise_dim))), cb).data
                    xr = xb + noise_sd * rng.normal(size=xb.shape)
                    xf = fake + noise_sd * rng.normal(size=fake.shape)
                    lr_ = self._disc(Tensor(xr), cb)
                    lf_ = self._disc(Tensor(xf), cb)
                    d_loss = bce_with_logits(lr_, np.full((m, 1), real_t)) + bce_with_logits(lf_, np.zeros((m, 1)))
                    optD.zero_grad()
                    d_loss.backward()
                    clip_grad_norm(optD.params, cfg.grad_clip or 5.0)
                    optD.step()
                fake = self._gen(Tensor(rng.normal(size=(m, net.noise_dim))), cb)
                g_loss = bce_with_logits(self._disc(fake + noise_sd * rng.normal(size=fake.shape), cb),
                                         np.ones((m, 1)))
                if self.moment_weight:
                    mf, sf, lf = self._moments(fake)
                    mr, sr, lr2 = self._moments(Tensor(xb))
                    mm = ((mf - mr) ** 2).mean() + ((sf - sr) ** 2).mean()
                    if lf is not None:
                        mm = mm + ((lf - lr2) ** 2).mean()
                    g_loss = g_loss + mm * self.moment_weight
                optG.zero_grad()
                # discriminator grads from this pass are discarded
                g_loss.backward()
                clip_grad_norm(optG.params, cfg.grad_clip or 5.0)
                optG.step()
                optD.zero_grad()
                g_tot += float(g_loss.data)
                d_tot += float(d_loss.data)
                nb += 1
            hist.train_loss.append(g_tot / max(nb, 1))
            hist.val_loss.append(d_tot / max(nb, 1))  # discriminator loss
            if cfg.verbose and (epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1):
                print(f"epoch {epoch:4d}  G={hist.train_loss[-1]:.4f}  D={hist.val_loss[-1]:.4f}")
        net.eval()
        self.history_ = hist

    def _fit(self, X, cond):
        self._build()
        self.history_ = None
        self._run(X, cond, None)

    def partial_fit(self, X, cond=None, epochs: Optional[int] = None):
        if not getattr(self, "is_fitted_", False):
            return self.fit(X, cond)
        cond = self._check_cond(cond, len(X))
        self._run(np.asarray(X, float), cond, epochs or max(20, (self.epochs or 100) // 5))
        return self

    def _sample(self, n, cond, rng):
        with no_grad():
            z = Tensor(rng.normal(size=(n, self.net_.noise_dim)))
            return self._gen(z, None if cond is None else Tensor(cond)).data

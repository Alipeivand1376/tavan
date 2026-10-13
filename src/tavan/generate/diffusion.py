"""Denoising diffusion probabilistic model (DDPM) with DDIM sampling."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..nn import Linear, Module, ResidualBlock, Tensor, mse, no_grad, sinusoidal_embedding, train
from ..registry import register
from .base import NeuralGenerator


def cosine_schedule(T: int, s: float = 0.008) -> np.ndarray:
    """Cumulative signal rates ``alpha_bar_t`` of Nichol & Dhariwal (2021)."""
    t = np.arange(T + 1) / T
    f = np.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    betas = np.clip(1 - ab[1:] / ab[:-1], 1e-5, 0.999)
    return np.cumprod(1 - betas)


def linear_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> np.ndarray:
    return np.cumprod(1 - np.linspace(beta_start, beta_end, T))


class _Denoiser(Module):
    def __init__(self, d, k, hidden, n_blocks, t_dim, dropout, rng):
        self.t_dim = t_dim
        self.k = k
        self.inp = Linear(d, hidden, rng=rng)
        self.temb = Linear(t_dim, hidden, rng=rng)
        self.cemb = Linear(k + 1, hidden, rng=rng) if k else None
        self.blocks = [ResidualBlock(hidden, 2 * hidden, hidden, dropout=dropout, rng=rng) for _ in range(n_blocks)]
        self.out = Linear(hidden, d, init="zeros", rng=rng)

    def forward(self, x, t_emb, c=None):
        h = self.inp(x) + self.temb(t_emb).silu()
        if self.cemb is not None and c is not None:
            h = h + self.cemb(c)
        for b in self.blocks:
            h = b(h)
        return self.out(h.silu())


@register("generator", "diffusion", aliases=("ddpm", "ddim"))
class Diffusion(NeuralGenerator):
    """Conditional denoising diffusion model for time-series profiles.

    Training follows DDPM (Ho et al., 2020) with the cosine noise schedule
    (Nichol & Dhariwal, 2021).  Sampling uses deterministic or stochastic
    DDIM (Song et al., 2021) with far fewer steps than training.  With
    conditions, classifier-free guidance (Ho & Salimans, 2022) is supported:
    conditions are randomly dropped during training (``cond_dropout``) and
    ``guidance`` > 0 sharpens conditional samples.

    Parameters
    ----------
    timesteps:
        Number of diffusion steps used in training.
    sample_steps:
        Number of DDIM steps at sampling time.
    eta:
        DDIM stochasticity (0 = deterministic ODE, 1 ≈ DDPM).
    hidden, n_blocks:
        Width and depth of the residual MLP denoiser.
    schedule:
        ``"cosine"`` or ``"linear"``.
    cond_dropout:
        Probability of dropping conditions during training.
    guidance:
        Classifier-free guidance weight ``w`` (``eps = (1+w) eps_c - w eps_u``).
    prediction:
        Network target: ``"v"`` (velocity, Salimans & Ho 2022 — well
        conditioned at all noise levels, default) or ``"eps"`` (noise).
    t_repeats:
        Each sample in a mini-batch is noised at this many random time-steps,
        which gives more gradient signal per optimisation step on CPU.
    clip_sample:
        Clip predicted clean samples to ``[-clip_sample, clip_sample]``
        (data are standardised, so 5 is a loose bound); ``None`` disables.
    """

    _module_attrs = ("net_",)
    _state_attrs = ()

    def __init__(self, timesteps: int = 200, sample_steps: int = 50, eta: float = 0.0, hidden: int = 256,
                 n_blocks: int = 3, t_dim: int = 64, schedule: str = "cosine", dropout: float = 0.0,
                 cond_dropout: float = 0.1, guidance: float = 0.0, prediction: str = "v", t_repeats: int = 4,
                 clip_sample: Optional[float] = 5.0,
                 epochs: Optional[int] = 400, lr: Optional[float] = 1e-3, batch_size: Optional[int] = 64,
                 seed: Optional[int] = 0, verbose: Optional[int] = 0, train=None) -> None:
        self.timesteps = timesteps
        self.sample_steps = sample_steps
        self.eta = eta
        self.hidden = hidden
        self.n_blocks = n_blocks
        self.t_dim = t_dim
        self.schedule = schedule
        self.dropout = dropout
        self.cond_dropout = cond_dropout
        self.guidance = guidance
        self.prediction = prediction
        self.t_repeats = t_repeats
        self.clip_sample = clip_sample
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose
        self.train = train

    # ---------------------------------------------------------------- setup
    def _alpha_bar(self) -> np.ndarray:
        if self.schedule == "cosine":
            return cosine_schedule(self.timesteps)
        if self.schedule == "linear":
            return linear_schedule(self.timesteps)
        raise ValueError("schedule must be 'cosine' or 'linear'")

    def _build(self):
        rng = np.random.default_rng(self.seed)
        self.net_ = _Denoiser(self.n_features_, self.n_cond_, self.hidden, self.n_blocks, self.t_dim,
                              self.dropout, rng)
        self._ab = self._alpha_bar()
        self._temb_table = sinusoidal_embedding(np.arange(self.timesteps) * (1000.0 / self.timesteps), self.t_dim)

    def _cond_input(self, cond: Optional[np.ndarray], keep: Optional[np.ndarray] = None):
        if cond is None:
            return None
        if keep is None:
            keep = np.ones((cond.shape[0], 1))
        return Tensor(np.concatenate([cond * keep, keep], axis=1))

    def _loss(self, net, xb, cb, rng=None, epoch=0):
        r = max(1, int(self.t_repeats))
        if r > 1:
            xb = np.tile(xb, (r, 1))
            cb = None if cb is None else np.tile(cb, (r, 1))
        n = xb.shape[0]
        t = rng.integers(0, self.timesteps, size=n)
        ab = self._ab[t][:, None]
        eps = rng.normal(size=xb.shape)
        xt = np.sqrt(ab) * xb + np.sqrt(1 - ab) * eps
        keep = None
        if cb is not None:
            keep = (rng.random((n, 1)) >= self.cond_dropout).astype(float) if net.training else np.ones((n, 1))
        pred = net(Tensor(xt), Tensor(self._temb_table[t]), self._cond_input(cb, keep))
        target = eps if self.prediction == "eps" else np.sqrt(ab) * eps - np.sqrt(1 - ab) * xb
        return mse(pred, target)

    def _fit(self, X, cond):
        self._build()
        self.history_ = train(self.net_, self._loss, [X, cond], self._train_config())

    def partial_fit(self, X, cond=None, epochs: Optional[int] = None):
        if not getattr(self, "is_fitted_", False):
            return self.fit(X, cond)
        X = np.asarray(X, dtype=float)
        cond = self._check_cond(cond, len(X))
        cfg = self._train_config(epochs=epochs or max(20, (self.epochs or 100) // 5),
                                 lr=(self.lr or 1e-3) * 0.3)
        self.history_ = train(self.net_, self._loss, [X, cond], cfg)
        return self

    # ------------------------------------------------------------- sampling
    def _net_out(self, x, t: int, cond):
        temb = Tensor(np.repeat(self._temb_table[t][None], x.shape[0], axis=0))
        if cond is None:
            return self.net_(Tensor(x), temb).data
        e_c = self.net_(Tensor(x), temb, self._cond_input(cond)).data
        if not self.guidance:
            return e_c
        e_u = self.net_(Tensor(x), temb, self._cond_input(cond, np.zeros((x.shape[0], 1)))).data
        return (1 + self.guidance) * e_c - self.guidance * e_u

    def _sample(self, n, cond, rng, return_trajectory: bool = False):
        steps = int(min(self.sample_steps, self.timesteps))
        ts = np.unique(np.linspace(0, self.timesteps - 1, steps).round().astype(int))[::-1]
        x = rng.normal(size=(n, self.n_features_))
        traj = [x.copy()] if return_trajectory else None
        self.net_.eval()
        with no_grad():
            for i, t in enumerate(ts):
                ab_t = self._ab[t]
                ab_prev = self._ab[ts[i + 1]] if i + 1 < len(ts) else 1.0
                out = self._net_out(x, t, cond)
                if self.prediction == "eps":
                    eps = out
                    x0 = (x - np.sqrt(1 - ab_t) * eps) / np.sqrt(ab_t)
                else:
                    x0 = np.sqrt(ab_t) * x - np.sqrt(1 - ab_t) * out
                    eps = np.sqrt(1 - ab_t) * x + np.sqrt(ab_t) * out
                if self.clip_sample:
                    x0 = np.clip(x0, -self.clip_sample, self.clip_sample)
                    eps = (x - np.sqrt(ab_t) * x0) / np.sqrt(1 - ab_t)
                sigma = self.eta * np.sqrt(max((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev), 0.0))
                dir_x = np.sqrt(max(1 - ab_prev - sigma ** 2, 0.0)) * eps
                x = np.sqrt(ab_prev) * x0 + dir_x
                if sigma > 0:
                    x = x + sigma * rng.normal(size=x.shape)
                if return_trajectory:
                    traj.append(x.copy())
        return (x, np.stack(traj)) if return_trajectory else x

    def sample_trajectory(self, n: int, cond=None, seed: Optional[int] = None):
        """Return samples and the full denoising trajectory ``(steps+1, n, d)``."""
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        if cond is not None:
            cond = np.atleast_2d(np.asarray(cond, float))
            if cond.shape[0] == 1:
                cond = np.repeat(cond, n, 0)
        return self._sample(n, cond, rng, return_trajectory=True)

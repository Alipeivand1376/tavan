"""Optimisers and learning-rate schedules."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np

from .layers import Parameter


def clip_grad_norm(params: Sequence[Parameter], max_norm: float) -> float:
    """Rescale gradients so that their global L2 norm is at most ``max_norm``.

    Returns the norm before clipping.
    """
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(np.sum(p.grad * p.grad))
    norm = math.sqrt(total)
    if max_norm and norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * scale
    return norm


class Optimizer:
    """Base optimiser."""

    def __init__(self, params: Sequence[Parameter], lr: float) -> None:
        self.params: List[Parameter] = [p for p in params]
        if not self.params:
            raise ValueError("optimizer received no parameters")
        self.lr = float(lr)
        self.base_lr = float(lr)
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent with (Nesterov) momentum."""

    def __init__(self, params, lr: float = 1e-2, momentum: float = 0.9,
                 nesterov: bool = False, weight_decay: float = 0.0) -> None:
        super().__init__(params, lr)
        self.momentum = momentum
        self.nesterov = nesterov
        self.wd = weight_decay
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        for p, v in zip(self.params, self.v):
            if p.grad is None:
                continue
            g = p.grad + self.wd * p.data if self.wd else p.grad
            v *= self.momentum
            v += g
            upd = g + self.momentum * v if self.nesterov else v
            p.data = p.data - self.lr * upd


class Adam(Optimizer):
    """Adam / AdamW optimiser.

    ``decoupled=True`` (default) applies AdamW-style weight decay.
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, decoupled: bool = True) -> None:
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.decoupled = decoupled
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.b1, self.b2
        c1 = 1.0 - b1 ** self.t
        c2 = 1.0 - b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            if self.wd and not self.decoupled:
                g = g + self.wd * p.data
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * (g * g)
            update = (m / c1) / (np.sqrt(v / c2) + self.eps)
            if self.wd and self.decoupled:
                update = update + self.wd * p.data
            p.data = p.data - self.lr * update


AdamW = Adam


class LRScheduler:
    """Base class: call :meth:`step` once per epoch."""

    def __init__(self, optimizer: Optimizer) -> None:
        self.opt = optimizer
        self.epoch = 0

    def get_lr(self) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def step(self, metric: Optional[float] = None) -> None:
        self.epoch += 1
        self.opt.lr = self.get_lr()


class CosineSchedule(LRScheduler):
    """Cosine annealing with linear warm-up."""

    def __init__(self, optimizer: Optimizer, total_epochs: int, warmup: int = 0,
                 min_lr_ratio: float = 0.05) -> None:
        super().__init__(optimizer)
        self.total = max(1, int(total_epochs))
        self.warmup = int(warmup)
        self.min_ratio = min_lr_ratio
        if self.warmup:
            optimizer.lr = optimizer.base_lr / (self.warmup + 1)

    def get_lr(self) -> float:
        e, base = self.epoch, self.opt.base_lr
        if self.warmup and e < self.warmup:
            return base * (e + 1) / (self.warmup + 1)
        progress = min(1.0, (e - self.warmup) / max(1, self.total - self.warmup))
        return base * (self.min_ratio + (1 - self.min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


class StepSchedule(LRScheduler):
    """Multiply the learning rate by ``gamma`` every ``step_size`` epochs."""

    def __init__(self, optimizer: Optimizer, step_size: int = 50, gamma: float = 0.5) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self) -> float:
        return self.opt.base_lr * self.gamma ** (self.epoch // self.step_size)


class ConstantSchedule(LRScheduler):
    def get_lr(self) -> float:
        return self.opt.lr


def make_optimizer(name: str, params, lr: float, weight_decay: float = 0.0, **kw) -> Optimizer:
    """Build an optimiser by name (``"adam"``, ``"adamw"``, ``"sgd"``)."""
    name = name.lower()
    if name in ("adam", "adamw"):
        return Adam(params, lr=lr, weight_decay=weight_decay, decoupled=(name == "adamw"), **kw)
    if name == "sgd":
        return SGD(params, lr=lr, weight_decay=weight_decay, **kw)
    raise ValueError(f"unknown optimizer {name!r}")


def make_scheduler(name: Optional[str], optimizer: Optimizer, epochs: int, warmup: int = 0) -> LRScheduler:
    if name in (None, "none", "constant"):
        return ConstantSchedule(optimizer)
    if name == "cosine":
        return CosineSchedule(optimizer, epochs, warmup=warmup)
    if name == "step":
        return StepSchedule(optimizer, step_size=max(1, epochs // 3))
    raise ValueError(f"unknown scheduler {name!r}")

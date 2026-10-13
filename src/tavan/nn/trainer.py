"""Generic mini-batch training loop with early stopping."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..config import TrainConfig
from .autograd import Tensor, no_grad
from .layers import Module
from .optim import clip_grad_norm, make_optimizer, make_scheduler

logger = logging.getLogger("tavan")

LossFn = Callable[..., Tensor]


@dataclass
class History:
    """Record of a training run."""

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val: float = math.inf
    seconds: float = 0.0
    stopped_early: bool = False

    def to_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "lr": self.lr,
            "best_epoch": self.best_epoch,
            "best_val": self.best_val,
            "seconds": self.seconds,
            "stopped_early": self.stopped_early,
        }

    def plot(self, ax=None):
        """Plot training/validation curves (requires matplotlib)."""
        from ..viz import plot_history

        return plot_history(self, ax=ax)


def _split(n: int, val_fraction: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    idx = rng.permutation(n)
    n_val = int(round(n * val_fraction)) if val_fraction > 0 else 0
    if n_val < 1 or n - n_val < 1:
        return idx, idx[:0]
    return idx[n_val:], idx[:n_val]


def train(
    model: Module,
    loss_fn: LossFn,
    arrays: Sequence[Optional[np.ndarray]],
    config: TrainConfig,
    val_arrays: Optional[Sequence[Optional[np.ndarray]]] = None,
    history: Optional[History] = None,
    extra_params: Sequence = (),
) -> History:
    """Train ``model`` by minimising ``loss_fn(model, *batch, rng=rng, epoch=e)``.

    Parameters
    ----------
    model:
        Module whose parameters are optimised.
    loss_fn:
        Callable returning a scalar :class:`Tensor`.  It receives the batch
        arrays positionally plus keyword arguments ``rng`` and ``epoch``.
    arrays:
        Training arrays sharing the first dimension (``None`` entries are
        passed through untouched).
    config:
        Optimisation settings.
    val_arrays:
        Explicit validation arrays.  When omitted, ``config.val_fraction`` of
        the training samples are held out.
    history:
        Existing history to append to (used for fine-tuning).
    """
    rng = np.random.default_rng(config.seed)
    arrays = list(arrays)
    n = next(a.shape[0] for a in arrays if a is not None)
    if val_arrays is None:
        tr_idx, va_idx = _split(n, config.val_fraction if config.patience or config.val_fraction else 0.0, rng)
        train_arrays = [a[tr_idx] if a is not None else None for a in arrays]
        val_arrays = [a[va_idx] if a is not None else None for a in arrays] if len(va_idx) else None
    else:
        train_arrays = arrays
        val_arrays = list(val_arrays)
    n_tr = next(a.shape[0] for a in train_arrays if a is not None)

    params = model.parameters() + list(extra_params)
    opt = make_optimizer(config.optimizer, params, config.lr, config.weight_decay)
    sched = make_scheduler(config.scheduler, opt, config.epochs, config.warmup)
    hist = history or History()
    best_state = None
    best = math.inf
    wait = 0
    bs = max(1, min(config.batch_size, n_tr))
    t0 = time.perf_counter()

    for epoch in range(config.epochs):
        model.train()
        perm = rng.permutation(n_tr)
        total, count = 0.0, 0
        for start in range(0, n_tr, bs):
            bi = perm[start:start + bs]
            batch = [a[bi] if a is not None else None for a in train_arrays]
            loss = loss_fn(model, *batch, rng=rng, epoch=epoch)
            opt.zero_grad()
            loss.backward()
            if config.grad_clip:
                clip_grad_norm(params, config.grad_clip)
            opt.step()
            lv = float(loss.data)
            if not math.isfinite(lv):
                raise FloatingPointError(
                    f"training diverged at epoch {epoch} (loss={lv}); try a lower lr or grad_clip"
                )
            total += lv * len(bi)
            count += len(bi)
        tr_loss = total / max(count, 1)
        hist.train_loss.append(tr_loss)
        hist.lr.append(opt.lr)

        monitor = tr_loss
        if val_arrays is not None:
            model.eval()
            with no_grad():
                vrng = np.random.default_rng(12345)
                vl = float(loss_fn(model, *val_arrays, rng=vrng, epoch=epoch).data)
            hist.val_loss.append(vl)
            monitor = vl

        improved = (not math.isfinite(best)) or monitor < best - config.min_delta * max(abs(best), 1e-8)
        if improved:
            best = monitor
            hist.best_epoch = len(hist.train_loss) - 1
            hist.best_val = best
            wait = 0
            if config.patience:
                best_state = model.state_dict()
        else:
            wait += 1
        sched.step(monitor)

        if config.verbose and (epoch % config.log_every == 0 or epoch == config.epochs - 1):
            msg = f"epoch {epoch:4d}  train={tr_loss:.5f}"
            if hist.val_loss:
                msg += f"  val={hist.val_loss[-1]:.5f}"
            msg += f"  lr={opt.lr:.2e}"
            print(msg)
            logger.info(msg)

        if config.patience and wait >= config.patience:
            hist.stopped_early = True
            if config.verbose:
                print(f"early stopping at epoch {epoch} (best epoch {hist.best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    hist.seconds += time.perf_counter() - t0
    return hist

"""Plotting helpers (matplotlib is an optional dependency).

Every function imports matplotlib lazily and raises a clear message if it is
missing, so the core library has no plotting dependency.
"""

from __future__ import annotations

import numpy as np


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plotting requires matplotlib; install tavan[plot] or pip install matplotlib") from exc
    return plt


def _ax(ax):
    if ax is not None:
        return ax.figure, ax
    plt = _plt()
    return plt.subplots(figsize=(9, 4))


def plot_series(ts, ax=None, max_points: int = 5000, **kwargs):
    """Plot a :class:`TimeSeries`, marking imputed/missing regions."""
    fig, ax = _ax(ax)
    x = ts.timestamps if ts.timestamps is not None else np.arange(ts.n_steps)
    V = ts.values
    step = max(1, ts.n_steps // max_points)
    for c in range(ts.n_channels):
        ax.plot(x[::step], V[::step, c], label=ts.names[c], lw=1.0, **kwargs)
    miss = ~np.isfinite(V).all(1)
    if miss.any():
        for i in np.where(miss)[0][::step]:
            ax.axvspan(x[max(i - 1, 0)], x[min(i + 1, ts.n_steps - 1)], color="red", alpha=0.08)
    ax.set_ylabel(ts.unit or "value")
    if ts.n_channels > 1:
        ax.legend(loc="best", fontsize=8)
    ax.set_title("Time series")
    return ax


def plot_history(history, ax=None):
    """Plot training and validation loss curves."""
    if history is None:
        raise ValueError("no training history available")
    fig, ax = _ax(ax)
    ax.plot(history.train_loss, label="train")
    if history.val_loss:
        ax.plot(history.val_loss, label="val")
    if history.best_epoch >= 0:
        ax.axvline(history.best_epoch, color="gray", ls="--", lw=0.8, label="best")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)
    ax.set_title("Training history")
    return ax


def plot_scenarios(scen, reference=None, ax=None, channel: int = 0, max_paths: int = 40, levels=(0.5, 0.9)):
    """Fan chart: quantile bands plus a sample of individual scenario paths."""
    fig, ax = _ax(ax)
    H = scen.horizon
    x = np.arange(H) if scen.timestamps is None else scen.timestamps
    V = scen.values[:, :, channel]
    colors = ["#c7d9f2", "#9ec1e8"]
    for i, lvl in enumerate(sorted(levels, reverse=True)):
        lo, hi = (1 - lvl) / 2, (1 + lvl) / 2
        qlo = np.quantile(V, lo, axis=0)
        qhi = np.quantile(V, hi, axis=0)
        ax.fill_between(x, qlo, qhi, color=colors[i % len(colors)], alpha=0.8,
                        label=f"{int(lvl * 100)}% band")
    rng = np.random.default_rng(0)
    for i in rng.choice(scen.n_scenarios, min(max_paths, scen.n_scenarios), replace=False):
        ax.plot(x, V[i], color="#2a6099", lw=0.4, alpha=0.25)
    ax.plot(x, scen.mean()[:, channel], color="#08306b", lw=2, label="mean")
    if reference is not None:
        ref = reference.values if hasattr(reference, "values") else np.asarray(reference)
        ref = ref.reshape(-1, H, scen.n_channels)[: , :, channel] if ref.ndim == 3 else ref.reshape(-1, H)
        ax.plot(x, ref[:5].T, color="black", lw=1.0, alpha=0.6)
    ax.set_title(f"Scenarios — {scen.names[channel]}")
    ax.set_ylabel(scen.unit or "value")
    ax.legend(fontsize=8, loc="best")
    return ax


def plot_forecast(result, history=None, ax=None, channel: int = 0, level: float = 0.9):
    """Plot a forecast with its prediction interval and optional history."""
    fig, ax = _ax(ax)
    H = result.horizon
    mean = result.mean if result.mean.ndim == 1 else result.mean[:, channel]
    x = np.arange(H) if result.timestamps is None else result.timestamps
    if history is not None:
        h = np.asarray(history)
        hx = np.arange(-len(h), 0) if result.timestamps is None else None
        ax.plot(hx if hx is not None else np.arange(len(h)), h if h.ndim == 1 else h[:, channel],
                color="black", lw=1, label="history")
    ax.plot(x, mean, color="#08519c", lw=2, label="forecast")
    if result.quantiles:
        try:
            lo, hi = result.interval(level)
            lo = lo if lo.ndim == 1 else lo[:, channel]
            hi = hi if hi.ndim == 1 else hi[:, channel]
            ax.fill_between(x, lo, hi, color="#9ecae1", alpha=0.6, label=f"{int(level * 100)}% PI")
        except Exception:
            pass
    ax.set_title("Forecast")
    ax.legend(fontsize=8)
    return ax


def plot_profiles(profiles, ax=None, max_lines: int = 100, color="#2a6099"):
    """Overlay many periodic profiles (e.g. daily load shapes)."""
    fig, ax = _ax(ax)
    P = np.asarray(profiles)
    if P.ndim == 3:
        P = P[:, :, 0]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(P), min(max_lines, len(P)), replace=False)
    ax.plot(P[idx].T, color=color, lw=0.4, alpha=0.3)
    ax.plot(P.mean(0), color="#08306b", lw=2, label="mean")
    ax.legend(fontsize=8)
    ax.set_title("Profiles")
    return ax

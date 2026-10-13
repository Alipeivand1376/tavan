"""A compact, vectorised reverse-mode automatic differentiation engine.

The engine operates on NumPy arrays and records a dynamic computation graph.
Every operation stores a closure that propagates gradients to its parents;
:meth:`Tensor.backward` walks the graph in reverse topological order.

The design goal is clarity and low overhead on CPU rather than generality:
all operations are batched NumPy calls, so the Python overhead per training
step is proportional to the number of *layers*, not the number of elements.

Example
-------
>>> import numpy as np
>>> from tavan.nn import Tensor
>>> w = Tensor(np.ones((3, 1)), requires_grad=True)
>>> x = Tensor(np.arange(6.0).reshape(2, 3))
>>> loss = ((x @ w) ** 2).mean()
>>> loss.backward()
>>> w.grad.shape
(3, 1)
"""

from __future__ import annotations

import contextlib
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, float, int, Sequence]

_GRAD_ENABLED = True
DTYPE = np.float64


@contextlib.contextmanager
def no_grad():
    """Context manager that disables graph construction (inference mode)."""
    global _GRAD_ENABLED
    previous = _GRAD_ENABLED
    _GRAD_ENABLED = False
    try:
        yield
    finally:
        _GRAD_ENABLED = previous


def is_grad_enabled() -> bool:
    """Return ``True`` when operations currently record a graph."""
    return _GRAD_ENABLED


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` over broadcast dimensions so that it matches ``shape``."""
    if grad.shape == shape:
        return grad
    ndim_extra = grad.ndim - len(shape)
    if ndim_extra > 0:
        grad = grad.sum(axis=tuple(range(ndim_extra)))
    axes = tuple(i for i, s in enumerate(shape) if s == 1 and grad.shape[i] != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return grad.reshape(shape)


def _as_array(x: ArrayLike) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x if x.dtype == DTYPE else x.astype(DTYPE)
    return np.asarray(x, dtype=DTYPE)


class Tensor:
    """N-dimensional array that records operations for differentiation.

    Parameters
    ----------
    data:
        Array-like content; converted to ``float64``.
    requires_grad:
        Whether gradients should be accumulated into :attr:`grad`.
    """

    __slots__ = ("data", "grad", "requires_grad", "_parents", "_backward", "name")
    __array_priority__ = 1000  # make ndarray <op> Tensor dispatch to Tensor

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        _parents: Tuple["Tensor", ...] = (),
        _backward: Optional[Callable[[np.ndarray], None]] = None,
        name: str = "",
    ) -> None:
        self.data = _as_array(data)
        self.grad: Optional[np.ndarray] = None
        self.requires_grad = bool(requires_grad)
        self._parents = _parents
        self._backward = _backward
        self.name = name

    # ------------------------------------------------------------------ utils
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def numpy(self) -> np.ndarray:
        """Return the underlying array (no copy)."""
        return self.data

    def item(self) -> float:
        return float(self.data.reshape(-1)[0]) if self.data.size == 1 else float("nan")

    def detach(self) -> "Tensor":
        """Return a new tensor sharing data but cut from the graph."""
        return Tensor(self.data)

    def zero_grad(self) -> None:
        self.grad = None

    def __repr__(self) -> str:
        rg = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor(shape={self.shape}{rg})"

    def __len__(self) -> int:
        return len(self.data)

    # -------------------------------------------------------------- plumbing
    @staticmethod
    def _make(data: np.ndarray, parents: Tuple["Tensor", ...], backward) -> "Tensor":
        needs = _GRAD_ENABLED and any(p.requires_grad for p in parents)
        if not needs:
            return Tensor(data)
        return Tensor(data, requires_grad=True, _parents=parents, _backward=backward)

    def _accum(self, g: np.ndarray) -> None:
        if not self.requires_grad:
            return
        if self.grad is None:
            self.grad = np.array(g, dtype=DTYPE, copy=True)
        else:
            self.grad = self.grad + g

    def backward(self, grad: Optional[ArrayLike] = None) -> None:
        """Back-propagate gradients from this tensor through the graph."""
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor that does not require grad")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be given for non-scalar outputs")
            grad = np.ones_like(self.data)
        grad = _as_array(grad)

        # iterative topological sort (no recursion limit issues)
        order = []
        visited = set()
        stack = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if p.requires_grad and id(p) not in visited:
                    stack.append((p, False))

        grads = {id(self): grad}
        for node in reversed(order):
            g = grads.pop(id(node), None)
            if g is None:
                continue
            if node._backward is None:  # leaf
                node._accum(g)
            else:
                for parent, pg in node._backward(g):
                    if pg is None or not parent.requires_grad:
                        continue
                    key = id(parent)
                    if key in grads:
                        grads[key] = grads[key] + pg
                    else:
                        grads[key] = pg
        # free graph references for non-leaf nodes to release memory
        for node in order:
            if node._backward is not None:
                node._parents = ()
                node._backward = None

    # ---------------------------------------------------------- arithmetic
    def __add__(self, other: ArrayLike) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        a, b = self, other

        def bw(g):
            return ((a, _unbroadcast(g, a.shape)), (b, _unbroadcast(g, b.shape)))

        return Tensor._make(a.data + b.data, (a, b), bw)

    __radd__ = __add__

    def __neg__(self) -> "Tensor":
        a = self
        return Tensor._make(-a.data, (a,), lambda g: ((a, -g),))

    def __sub__(self, other: ArrayLike) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        a, b = self, other

        def bw(g):
            return ((a, _unbroadcast(g, a.shape)), (b, _unbroadcast(-g, b.shape)))

        return Tensor._make(a.data - b.data, (a, b), bw)

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        return Tensor(other) - self

    def __mul__(self, other: ArrayLike) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        a, b = self, other

        def bw(g):
            return (
                (a, _unbroadcast(g * b.data, a.shape) if a.requires_grad else None),
                (b, _unbroadcast(g * a.data, b.shape) if b.requires_grad else None),
            )

        return Tensor._make(a.data * b.data, (a, b), bw)

    __rmul__ = __mul__

    def __truediv__(self, other: ArrayLike) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        a, b = self, other

        def bw(g):
            return (
                (a, _unbroadcast(g / b.data, a.shape) if a.requires_grad else None),
                (b, _unbroadcast(-g * a.data / (b.data ** 2), b.shape) if b.requires_grad else None),
            )

        return Tensor._make(a.data / b.data, (a, b), bw)

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        return Tensor(other) / self

    def __pow__(self, p: float) -> "Tensor":
        if isinstance(p, Tensor):
            raise TypeError("only scalar exponents are supported")
        a = self
        p = float(p)
        return Tensor._make(a.data ** p, (a,), lambda g: ((a, g * p * a.data ** (p - 1.0)),))

    def __matmul__(self, other: ArrayLike) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        a, b = self, other
        out = a.data @ b.data

        def bw(g):
            ga = gb = None
            if a.requires_grad:
                if b.ndim == 1:
                    ga = np.multiply.outer(g, b.data)
                else:
                    ga = g @ np.swapaxes(b.data, -1, -2)
                ga = _unbroadcast(ga, a.shape)
            if b.requires_grad:
                if a.ndim == 1:
                    gb = np.multiply.outer(a.data, g)
                else:
                    gb = np.swapaxes(a.data, -1, -2) @ g
                gb = _unbroadcast(gb, b.shape)
            return ((a, ga), (b, gb))

        return Tensor._make(out, (a, b), bw)

    def __rmatmul__(self, other: ArrayLike) -> "Tensor":
        return Tensor(other) @ self

    # ---------------------------------------------------------- reductions
    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        a = self

        def bw(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            return ((a, np.broadcast_to(g, a.shape).copy()),)

        return Tensor._make(a.data.sum(axis=axis, keepdims=keepdims), (a,), bw)

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            n = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            n = int(np.prod([self.data.shape[ax] for ax in axes]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def var(self, axis=None, keepdims: bool = False) -> "Tensor":
        mu = self.mean(axis=axis, keepdims=True)
        return ((self - mu) ** 2).mean(axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims: bool = False) -> "Tensor":
        a = self
        out = a.data.max(axis=axis, keepdims=True)
        mask = (a.data == out).astype(DTYPE)
        mask /= mask.sum(axis=axis, keepdims=True)
        res = out if keepdims else (out.reshape(()) if axis is None else np.squeeze(out, axis))

        def bw(g):
            if not keepdims and axis is not None:
                g = np.expand_dims(g, axis)
            return ((a, mask * g),)

        return Tensor._make(res, (a,), bw)

    # ------------------------------------------------------------- shaping
    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        a = self
        return Tensor._make(a.data.reshape(shape), (a,), lambda g: ((a, g.reshape(a.shape)),))

    def transpose(self, *axes) -> "Tensor":
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        a = self
        if not axes:
            axes = tuple(reversed(range(a.ndim)))
        inv = tuple(np.argsort(axes))
        return Tensor._make(a.data.transpose(axes), (a,), lambda g: ((a, g.transpose(inv)),))

    def swapaxes(self, ax1: int, ax2: int) -> "Tensor":
        axes = list(range(self.ndim))
        axes[ax1], axes[ax2] = axes[ax2], axes[ax1]
        return self.transpose(tuple(axes))

    def __getitem__(self, idx) -> "Tensor":
        a = self
        if isinstance(idx, Tensor):
            idx = idx.data.astype(int)

        def bw(g):
            full = np.zeros_like(a.data)
            np.add.at(full, idx, g)
            return ((a, full),)

        return Tensor._make(a.data[idx], (a,), bw)

    # ---------------------------------------------------------- elementwise
    def exp(self) -> "Tensor":
        a = self
        out = np.exp(a.data)
        return Tensor._make(out, (a,), lambda g: ((a, g * out),))

    def log(self) -> "Tensor":
        a = self
        return Tensor._make(np.log(a.data), (a,), lambda g: ((a, g / a.data),))

    def sqrt(self) -> "Tensor":
        a = self
        out = np.sqrt(a.data)
        return Tensor._make(out, (a,), lambda g: ((a, g * 0.5 / np.maximum(out, 1e-12)),))

    def abs(self) -> "Tensor":
        a = self
        return Tensor._make(np.abs(a.data), (a,), lambda g: ((a, g * np.sign(a.data)),))

    def tanh(self) -> "Tensor":
        a = self
        out = np.tanh(a.data)
        return Tensor._make(out, (a,), lambda g: ((a, g * (1.0 - out ** 2)),))

    def sigmoid(self) -> "Tensor":
        a = self
        out = _stable_sigmoid(a.data)
        return Tensor._make(out, (a,), lambda g: ((a, g * out * (1.0 - out)),))

    def relu(self) -> "Tensor":
        a = self
        mask = a.data > 0
        return Tensor._make(a.data * mask, (a,), lambda g: ((a, g * mask),))

    def leaky_relu(self, slope: float = 0.2) -> "Tensor":
        a = self
        factor = np.where(a.data > 0, 1.0, slope)
        return Tensor._make(a.data * factor, (a,), lambda g: ((a, g * factor),))

    def gelu(self) -> "Tensor":
        """Gaussian error linear unit (tanh approximation)."""
        a = self
        x = a.data
        c = np.sqrt(2.0 / np.pi)
        inner = c * (x + 0.044715 * x ** 3)
        t = np.tanh(inner)
        out = 0.5 * x * (1.0 + t)

        def bw(g):
            dinner = c * (1.0 + 3 * 0.044715 * x ** 2)
            d = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * dinner
            return ((a, g * d),)

        return Tensor._make(out, (a,), bw)

    def silu(self) -> "Tensor":
        a = self
        s = _stable_sigmoid(a.data)
        out = a.data * s
        return Tensor._make(out, (a,), lambda g: ((a, g * (s + a.data * s * (1 - s))),))

    def softplus(self) -> "Tensor":
        a = self
        x = a.data
        out = np.logaddexp(0.0, x)
        return Tensor._make(out, (a,), lambda g: ((a, g * _stable_sigmoid(x)),))

    def clip(self, lo: Optional[float], hi: Optional[float]) -> "Tensor":
        a = self
        out = np.clip(a.data, lo, hi)
        mask = (out == a.data).astype(DTYPE)
        return Tensor._make(out, (a,), lambda g: ((a, g * mask),))

    def softmax(self, axis: int = -1) -> "Tensor":
        a = self
        z = a.data - a.data.max(axis=axis, keepdims=True)
        e = np.exp(z)
        s = e / e.sum(axis=axis, keepdims=True)

        def bw(g):
            return ((a, s * (g - (g * s).sum(axis=axis, keepdims=True))),)

        return Tensor._make(s, (a,), bw)

    def log_softmax(self, axis: int = -1) -> "Tensor":
        a = self
        z = a.data - a.data.max(axis=axis, keepdims=True)
        lse = np.log(np.exp(z).sum(axis=axis, keepdims=True))
        out = z - lse
        s = np.exp(out)

        def bw(g):
            return ((a, g - s * g.sum(axis=axis, keepdims=True)),)

        return Tensor._make(out, (a,), bw)

    # comparison helpers return plain arrays (non-differentiable)
    def __gt__(self, other):
        return self.data > (other.data if isinstance(other, Tensor) else other)

    def __lt__(self, other):
        return self.data < (other.data if isinstance(other, Tensor) else other)


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


# ---------------------------------------------------------------- functions
def tensor(data: ArrayLike, requires_grad: bool = False) -> Tensor:
    """Create a :class:`Tensor`."""
    return Tensor(data, requires_grad=requires_grad)


def concat(tensors: Iterable[Tensor], axis: int = -1) -> Tensor:
    """Concatenate tensors along ``axis`` (differentiable)."""
    ts = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
    out = np.concatenate([t.data for t in ts], axis=axis)
    sizes = [t.shape[axis] for t in ts]
    splits = np.cumsum(sizes)[:-1]

    def bw(g):
        parts = np.split(g, splits, axis=axis)
        return tuple((t, p) for t, p in zip(ts, parts))

    return Tensor._make(out, tuple(ts), bw)


def stack(tensors: Iterable[Tensor], axis: int = 0) -> Tensor:
    """Stack tensors along a new ``axis`` (differentiable)."""
    ts = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
    out = np.stack([t.data for t in ts], axis=axis)

    def bw(g):
        return tuple((t, np.take(g, i, axis=axis)) for i, t in enumerate(ts))

    return Tensor._make(out, tuple(ts), bw)


def where(cond: np.ndarray, a: Tensor, b: Tensor) -> Tensor:
    """Element-wise selection with a constant boolean mask."""
    a = a if isinstance(a, Tensor) else Tensor(a)
    b = b if isinstance(b, Tensor) else Tensor(b)
    cond = np.asarray(cond, dtype=bool)
    out = np.where(cond, a.data, b.data)

    def bw(g):
        return (
            (a, _unbroadcast(np.where(cond, g, 0.0), a.shape)),
            (b, _unbroadcast(np.where(cond, 0.0, g), b.shape)),
        )

    return Tensor._make(out, (a, b), bw)


def numerical_grad(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central finite-difference gradient, used for testing."""
    x = np.array(x, dtype=DTYPE)
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        fp = f(x)
        x[i] = old - eps
        fm = f(x)
        x[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g

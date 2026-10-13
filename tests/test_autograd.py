"""Gradient checks for the autograd engine against finite differences."""

import numpy as np
import pytest

from tavan.nn import Tensor, concat, numerical_grad, stack
from tavan.nn.layers import MLP, LayerNorm, MultiHeadAttention

rng = np.random.default_rng(0)


def _check(fn, *shapes, tol=1e-5):
    xs = [rng.normal(size=s) for s in shapes]
    ts = [Tensor(x, requires_grad=True) for x in xs]
    fn(*ts).backward()
    for i, x in enumerate(xs):
        def f(v, i=i):
            args = [Tensor(v if j == i else xs[j]) for j in range(len(xs))]
            return float(fn(*args).data)
        ng = numerical_grad(f, x.copy())
        assert np.max(np.abs(ng - ts[i].grad)) < tol, f"grad mismatch arg {i}"


@pytest.mark.parametrize("fn,shapes", [
    (lambda a, b: (a @ b).sum(), [(3, 4), (4, 2)]),
    (lambda a, b: ((a @ b) ** 2).mean(), [(2, 3, 4), (2, 4, 5)]),
    (lambda a, b: ((a + b) * (a - b)).sum(), [(3, 4), (4,)]),
    (lambda a, b: (a / (b * b + 1)).sum(), [(3, 4), (1, 4)]),
    (lambda a: (a.softmax(-1) * Tensor(np.arange(5.0))).sum(), [(3, 5)]),
    (lambda a: a.log_softmax(-1)[:, 1].sum(), [(3, 5)]),
    (lambda a: a.gelu().sum(), [(4, 4)]),
    (lambda a: a.silu().sum(), [(4, 4)]),
    (lambda a: a.softplus().sum(), [(4, 4)]),
    (lambda a: a.tanh().sum(), [(4, 4)]),
    (lambda a: (a.transpose(2, 0, 1).reshape(4, 6) ** 2).sum(), [(2, 3, 4)]),
    (lambda a: (a[1:, ::2] ** 2).sum(), [(4, 5)]),
    (lambda a, b: (concat([a, b], axis=1) ** 3).sum(), [(2, 3), (2, 4)]),
    (lambda a, b: (stack([a, b], axis=0)).sum(), [(2, 3), (2, 3)]),
    (lambda a: a.var(axis=0).sum(), [(5, 3)]),
    (lambda a: a.max(axis=1).sum(), [(4, 6)]),
    (lambda a: ((a * a + 1).log() + (a * 0.1).exp()).sum(), [(3, 3)]),
    (lambda a: (a * a + 1).sqrt().sum(), [(3, 3)]),
    (lambda a: a.clip(-0.5, 0.5).sum(), [(4, 4)]),
])
def test_elementwise_grads(fn, shapes):
    _check(fn, *shapes)


def test_layernorm_grad():
    ln = LayerNorm(4)
    x = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
    (ln(x) ** 2).sum().backward()

    def f(v):
        return float((ln(Tensor(v)) ** 2).sum().data)
    assert np.max(np.abs(numerical_grad(f, x.data.copy()) - x.grad)) < 1e-5


def test_attention_grad():
    mha = MultiHeadAttention(8, 2, rng=0)
    x = Tensor(rng.normal(size=(2, 5, 8)), requires_grad=True)
    (mha(x) ** 2).sum().backward()

    def f(v):
        return float((mha(Tensor(v)) ** 2).sum().data)
    assert np.max(np.abs(numerical_grad(f, x.data.copy()) - x.grad)) < 1e-4


def test_mlp_trains():
    from tavan.config import TrainConfig
    from tavan.nn import mse, train
    X = rng.normal(size=(400, 3))
    y = np.sin(X[:, :1]) + X[:, 1:2] * X[:, 2:3]
    net = MLP([3, 64, 64, 1], rng=0)
    h = train(net, lambda m, xb, yb, **k: mse(m(Tensor(xb)), yb), [X, y],
              TrainConfig(epochs=120, lr=3e-3, patience=0))
    assert h.train_loss[-1] < 0.05 * h.train_loss[0]


def test_no_grad_blocks_graph():
    from tavan.nn import no_grad
    x = Tensor(np.ones(3), requires_grad=True)
    with no_grad():
        y = (x * 2).sum()
    assert not y.requires_grad

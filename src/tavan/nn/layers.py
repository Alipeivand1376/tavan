"""Neural-network building blocks on top of :mod:`tavan.nn.autograd`."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from .autograd import Tensor, concat

Activation = Union[str, Callable[[Tensor], Tensor], None]


def get_activation(act: Activation) -> Callable[[Tensor], Tensor]:
    """Resolve an activation given by name or callable."""
    if act is None or act == "linear" or act == "identity":
        return lambda x: x
    if callable(act):
        return act
    table = {
        "relu": Tensor.relu,
        "gelu": Tensor.gelu,
        "tanh": Tensor.tanh,
        "sigmoid": Tensor.sigmoid,
        "silu": Tensor.silu,
        "swish": Tensor.silu,
        "softplus": Tensor.softplus,
        "leaky_relu": Tensor.leaky_relu,
    }
    try:
        return table[act]
    except KeyError:
        raise ValueError(f"unknown activation {act!r}; choose from {sorted(table)}") from None


class Parameter(Tensor):
    """A trainable tensor (``requires_grad=True`` by default)."""

    __slots__ = ()

    def __init__(self, data, name: str = "") -> None:
        super().__init__(data, requires_grad=True, name=name)


class Module:
    """Base class for all layers and models.

    Sub-modules and parameters assigned as attributes (or stored in lists)
    are discovered automatically by :meth:`parameters` and
    :meth:`named_parameters`.
    """

    training: bool = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover - abstract
        raise NotImplementedError

    # --------------------------------------------------------- traversal
    def _children(self) -> Iterator[Tuple[str, object]]:
        for name, value in vars(self).items():
            if isinstance(value, (Module, Parameter)):
                yield name, value
            elif isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    if isinstance(v, (Module, Parameter)):
                        yield f"{name}.{i}", v

    def named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        for name, value in self._children():
            full = f"{prefix}{name}"
            if isinstance(value, Parameter):
                yield full, value
            else:
                yield from value.named_parameters(prefix=full + ".")

    def parameters(self) -> List[Parameter]:
        return [p for _, p in self.named_parameters()]

    def modules(self) -> Iterator["Module"]:
        yield self
        for _, value in self._children():
            if isinstance(value, Module):
                yield from value.modules()

    def num_parameters(self) -> int:
        return int(sum(p.size for p in self.parameters()))

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def train(self, mode: bool = True) -> "Module":
        for m in self.modules():
            m.training = mode
        return self

    def eval(self) -> "Module":
        return self.train(False)

    # -------------------------------------------------------- state I/O
    def state_dict(self) -> "OrderedDict[str, np.ndarray]":
        return OrderedDict((n, p.data.copy()) for n, p in self.named_parameters())

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        params = dict(self.named_parameters())
        missing = set(params) - set(state)
        if missing:
            raise KeyError(f"missing parameters in state: {sorted(missing)[:5]}")
        for name, p in params.items():
            arr = np.asarray(state[name], dtype=p.data.dtype)
            if arr.shape != p.shape:
                raise ValueError(f"shape mismatch for {name}: {arr.shape} vs {p.shape}")
            p.data = arr.copy()


def _rng(seed_or_rng) -> np.random.Generator:
    if isinstance(seed_or_rng, np.random.Generator):
        return seed_or_rng
    return np.random.default_rng(seed_or_rng)


class Linear(Module):
    """Affine map ``y = x W + b`` with Kaiming/Xavier-style initialisation."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 init: str = "kaiming", rng=None) -> None:
        rng = _rng(rng)
        if init == "kaiming":
            std = np.sqrt(2.0 / in_features)
        elif init == "xavier":
            std = np.sqrt(2.0 / (in_features + out_features))
        elif init == "zeros":
            std = 0.0
        else:
            std = float(init)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(rng.normal(0.0, std, size=(in_features, out_features)))
        self.bias = Parameter(np.zeros(out_features)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


class LayerNorm(Module):
    """Layer normalisation over the last dimension."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        self.eps = eps
        self.gamma = Parameter(np.ones(dim))
        self.beta = Parameter(np.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        mu = x.mean(axis=-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(axis=-1, keepdims=True)
        return xc / (var + self.eps).sqrt() * self.gamma + self.beta


class Dropout(Module):
    """Inverted dropout; inactive in evaluation mode."""

    def __init__(self, p: float = 0.1, rng=None) -> None:
        self.p = float(p)
        self.rng = _rng(rng)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p <= 0.0:
            return x
        mask = (self.rng.random(x.shape) >= self.p) / (1.0 - self.p)
        return x * mask


class Sequential(Module):
    """Chain modules or callables."""

    def __init__(self, *layers) -> None:
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Lambda(Module):
    """Wrap a parameter-free function as a module."""

    def __init__(self, fn: Callable[[Tensor], Tensor]) -> None:
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x)


class MLP(Module):
    """Multi-layer perceptron.

    Parameters
    ----------
    sizes:
        ``[in, hidden_1, ..., out]``.
    activation:
        Activation for hidden layers.
    out_activation:
        Activation applied to the output layer (default: none).
    dropout:
        Dropout probability after each hidden activation.
    layer_norm:
        Insert :class:`LayerNorm` before each hidden activation.
    """

    def __init__(self, sizes: Sequence[int], activation: Activation = "gelu",
                 out_activation: Activation = None, dropout: float = 0.0,
                 layer_norm: bool = False, rng=None) -> None:
        rng = _rng(rng)
        if len(sizes) < 2:
            raise ValueError("MLP needs at least input and output sizes")
        self.linears = [Linear(a, b, rng=rng) for a, b in zip(sizes[:-1], sizes[1:])]
        self.norms = [LayerNorm(b) for b in sizes[1:-1]] if layer_norm else []
        self.dropouts = [Dropout(dropout, rng=rng) for _ in sizes[1:-1]] if dropout > 0 else []
        self.act = get_activation(activation)
        self.out_act = get_activation(out_activation)

    def forward(self, x: Tensor) -> Tensor:
        n = len(self.linears)
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < n - 1:
                if self.norms:
                    x = self.norms[i](x)
                x = self.act(x)
                if self.dropouts:
                    x = self.dropouts[i](x)
        return self.out_act(x)


class ResidualBlock(Module):
    """``x + Dropout(Linear(act(Linear(LN(x)))))`` with optional projection.

    Used by TiDE-style encoders and the diffusion denoiser.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0,
                 activation: Activation = "gelu", layer_norm: bool = True, rng=None) -> None:
        rng = _rng(rng)
        self.fc1 = Linear(in_dim, hidden, rng=rng)
        self.fc2 = Linear(hidden, out_dim, rng=rng)
        self.skip = Linear(in_dim, out_dim, rng=rng) if in_dim != out_dim else None
        self.norm = LayerNorm(out_dim) if layer_norm else None
        self.drop = Dropout(dropout, rng=rng)
        self.act = get_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        h = self.fc2(self.act(self.fc1(x)))
        h = self.drop(h)
        res = x if self.skip is None else self.skip(x)
        out = h + res
        return self.norm(out) if self.norm is not None else out


class MultiHeadAttention(Module):
    """Multi-head scaled dot-product self-attention.

    Input shape ``(batch, tokens, d_model)``.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0, rng=None) -> None:
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        rng = _rng(rng)
        self.h = n_heads
        self.dk = d_model // n_heads
        self.qkv = Linear(d_model, 3 * d_model, init="xavier", rng=rng)
        self.out = Linear(d_model, d_model, init="xavier", rng=rng)
        self.drop = Dropout(dropout, rng=rng)

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.h, self.dk).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (b, h, t, dk)
        scores = (q @ k.swapaxes(-1, -2)) * (1.0 / np.sqrt(self.dk))
        attn = self.drop(scores.softmax(axis=-1))
        ctx = (attn @ v).transpose(0, 2, 1, 3).reshape(b, t, d)
        return self.out(ctx)


class TransformerEncoderLayer(Module):
    """Pre-norm Transformer encoder layer (attention + feed-forward)."""

    def __init__(self, d_model: int, n_heads: int = 4, d_ff: Optional[int] = None,
                 dropout: float = 0.0, activation: Activation = "gelu", rng=None) -> None:
        rng = _rng(rng)
        d_ff = d_ff or 2 * d_model
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, rng=rng)
        self.norm2 = LayerNorm(d_model)
        self.ff = MLP([d_model, d_ff, d_model], activation=activation, rng=rng)
        self.drop = Dropout(dropout, rng=rng)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


def sinusoidal_embedding(t: np.ndarray, dim: int, max_period: float = 10000.0) -> np.ndarray:
    """Sinusoidal embedding of (possibly fractional) positions/time-steps."""
    t = np.asarray(t, dtype=np.float64).reshape(-1, 1)
    half = dim // 2
    freqs = np.exp(-np.log(max_period) * np.arange(half) / max(half, 1))
    args = t * freqs[None, :]
    emb = np.concatenate([np.sin(args), np.cos(args)], axis=1)
    if dim % 2:
        emb = np.concatenate([emb, np.zeros((emb.shape[0], 1))], axis=1)
    return emb


__all__ = [
    "Parameter", "Module", "Linear", "LayerNorm", "Dropout", "Sequential", "Lambda",
    "MLP", "ResidualBlock", "MultiHeadAttention", "TransformerEncoderLayer",
    "get_activation", "sinusoidal_embedding", "concat",
]

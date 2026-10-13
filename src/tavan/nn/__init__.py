"""``tavan.nn`` — a minimal deep-learning toolkit written on NumPy.

It provides reverse-mode automatic differentiation (:class:`Tensor`),
common layers, optimisers, losses and a training loop.  All generative and
deep forecasting models in ``tavan`` are built on it, so the package needs no
deep-learning framework and runs on any CPU.
"""

from .autograd import Tensor, concat, is_grad_enabled, no_grad, numerical_grad, stack, tensor, where
from .layers import (
    MLP,
    Dropout,
    Lambda,
    LayerNorm,
    Linear,
    Module,
    MultiHeadAttention,
    Parameter,
    ResidualBlock,
    Sequential,
    TransformerEncoderLayer,
    get_activation,
    sinusoidal_embedding,
)
from .losses import bce_with_logits, gaussian_nll, get_loss, huber, kl_standard_normal, mae, mse, pinball
from .optim import SGD, Adam, AdamW, CosineSchedule, StepSchedule, clip_grad_norm, make_optimizer
from .trainer import History, train

__all__ = [
    "Tensor", "tensor", "concat", "stack", "where", "no_grad", "is_grad_enabled", "numerical_grad",
    "Module", "Parameter", "Linear", "LayerNorm", "Dropout", "Sequential", "Lambda", "MLP",
    "ResidualBlock", "MultiHeadAttention", "TransformerEncoderLayer", "get_activation",
    "sinusoidal_embedding", "mse", "mae", "huber", "pinball", "gaussian_nll",
    "kl_standard_normal", "bce_with_logits", "get_loss", "SGD", "Adam", "AdamW",
    "CosineSchedule", "StepSchedule", "clip_grad_norm", "make_optimizer", "History", "train",
]

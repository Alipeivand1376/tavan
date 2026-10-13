"""Common estimator base class and safe (pickle-free) persistence.

Models are saved to a single ``.npz`` archive that contains NumPy arrays and
one JSON metadata record.  Loading never executes code from the file, unlike
``pickle``.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from ._version import __version__


class BaseEstimator:
    """Base class providing ``get_params``/``set_params``, ``repr`` and I/O.

    Sub-classes declare learned attributes in ``_state_attrs``; those are
    persisted by :meth:`save`.  Values may be arrays, JSON-serialisable
    objects, nested estimators or :class:`tavan.nn.Module` instances (saved as
    their ``state_dict``; sub-classes rebuild the module in :meth:`_set_state`).
    """

    _state_attrs: Tuple[str, ...] = ()

    # ------------------------------------------------------------ params
    @classmethod
    def _param_names(cls) -> Tuple[str, ...]:
        sig = inspect.signature(cls.__init__)
        return tuple(
            p.name for p in sig.parameters.values()
            if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        )

    def get_params(self) -> Dict[str, Any]:
        """Constructor arguments of this estimator."""
        return {n: getattr(self, n) for n in self._param_names() if hasattr(self, n)}

    def set_params(self, **params: Any) -> "BaseEstimator":
        """Set constructor arguments (does not refit)."""
        valid = set(self._param_names())
        for k, v in params.items():
            if k not in valid:
                raise ValueError(f"invalid parameter {k!r} for {type(self).__name__}; valid: {sorted(valid)}")
            setattr(self, k, v)
        return self

    def clone(self) -> "BaseEstimator":
        """Unfitted copy with identical parameters."""
        return type(self)(**self.get_params())

    def __repr__(self) -> str:
        defaults = {
            n: p.default for n, p in inspect.signature(type(self).__init__).parameters.items()
            if p.default is not inspect.Parameter.empty
        }
        parts = []
        for k, v in self.get_params().items():
            if k in defaults:
                try:
                    if defaults[k] is v or defaults[k] == v:
                        continue
                except Exception:  # array comparisons etc.
                    pass
            r = repr(v)
            parts.append(f"{k}={r if len(r) < 60 else r[:57] + '...'}")
        return f"{type(self).__name__}({', '.join(parts)})"

    # ------------------------------------------------------------- state
    def _get_state(self) -> Dict[str, Any]:
        return {a: getattr(self, a) for a in self._state_attrs if hasattr(self, a)}

    def _set_state(self, state: Dict[str, Any]) -> None:
        for k, v in state.items():
            setattr(self, k, v)

    # --------------------------------------------------------------- I/O
    def save(self, path: str) -> str:
        """Save the fitted estimator to ``path`` (``.npz`` appended if missing)."""
        if not str(path).endswith(".npz"):
            path = f"{path}.npz"
        arrays: Dict[str, np.ndarray] = {}
        meta = _encode_estimator(self, arrays, "e")
        meta["tavan_version"] = __version__
        arrays["__meta__"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
        np.savez_compressed(path, **arrays)
        return str(path)

    @classmethod
    def load(cls, path: str) -> "BaseEstimator":
        """Load an estimator saved with :meth:`save`."""
        return load(path)


# ----------------------------------------------------------------- codec
def _json_safe(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, tuple):
        return {"__tuple__": [_json_safe(x) for x in v]}
    if isinstance(v, list):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    return v


def _encode(v: Any, arrays: Dict[str, np.ndarray], prefix: str) -> Any:
    from .nn.layers import Module

    if isinstance(v, np.ndarray):
        if v.dtype == object:
            raise TypeError(f"cannot save object array at {prefix}")
        arrays[prefix] = v
        return {"__array__": prefix}
    if isinstance(v, BaseEstimator):
        return {"__estimator__": _encode_estimator(v, arrays, prefix)}
    from .config import TrainConfig

    if isinstance(v, TrainConfig):
        return {"__trainconfig__": _json_safe(v.to_dict())}
    if isinstance(v, Module):
        keys = []
        for name, arr in v.state_dict().items():
            k = f"{prefix}/{name}"
            arrays[k] = arr
            keys.append([name, k])
        return {"__module__": keys}
    if isinstance(v, dict):
        return {"__dict__": {str(k): _encode(x, arrays, f"{prefix}.{k}") for k, x in v.items()}}
    if isinstance(v, (list, tuple)):
        enc = [_encode(x, arrays, f"{prefix}[{i}]") for i, x in enumerate(v)]
        return {"__tuple__": enc} if isinstance(v, tuple) else {"__list__": enc}
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return _json_safe(v)
    if callable(v):
        name = getattr(v, "__name__", None)
        if name and getattr(v, "__module__", "").startswith(("numpy", "tavan", "builtins")):
            return {"__callable__": f"{v.__module__}:{v.__qualname__}"}
        raise TypeError(f"cannot save custom callable {v!r} at {prefix}")
    raise TypeError(f"cannot save value of type {type(v).__name__} at {prefix}")


def _encode_estimator(est: BaseEstimator, arrays: Dict[str, np.ndarray], prefix: str) -> Dict[str, Any]:
    from .registry import key_of

    params = {k: _encode(v, arrays, f"{prefix}.p.{k}") for k, v in est.get_params().items()}
    state = {k: _encode(v, arrays, f"{prefix}.s.{k}") for k, v in est._get_state().items()}
    return {"class": key_of(est), "params": params, "state": state}


def _decode(v: Any, arrays) -> Any:
    if isinstance(v, dict):
        if "__array__" in v:
            return np.array(arrays[v["__array__"]])
        if "__estimator__" in v:
            return _decode_estimator(v["__estimator__"], arrays)
        if "__module__" in v:
            return {name: np.array(arrays[k]) for name, k in v["__module__"]}
        if "__dict__" in v:
            return {k: _decode(x, arrays) for k, x in v["__dict__"].items()}
        if "__list__" in v:
            return [_decode(x, arrays) for x in v["__list__"]]
        if "__tuple__" in v:
            return tuple(_decode(x, arrays) for x in v["__tuple__"])
        if "__callable__" in v:
            return _resolve_callable(v["__callable__"])
        if "__trainconfig__" in v:
            from .config import TrainConfig

            return TrainConfig(**{k: _decode(x, arrays) for k, x in v["__trainconfig__"].items()})
        return {k: _decode(x, arrays) for k, x in v.items()}
    return v


def _resolve_callable(path: str):
    import importlib

    mod_name, qual = path.split(":", 1)
    if not mod_name.startswith(("numpy", "tavan", "builtins")):
        raise ValueError(f"refusing to import callable from {mod_name!r}")
    obj = importlib.import_module(mod_name)
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj


def _decode_estimator(meta: Dict[str, Any], arrays) -> BaseEstimator:
    from .registry import from_key

    cls = from_key(meta["class"])
    params = {k: _decode(v, arrays) for k, v in meta["params"].items()}
    obj = cls(**params)
    state = {k: _decode(v, arrays) for k, v in meta["state"].items()}
    obj._set_state(state)
    return obj


def load(path: str) -> BaseEstimator:
    """Load any ``tavan`` estimator saved with ``.save()``."""
    if not str(path).endswith(".npz"):
        path = f"{path}.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    meta = json.loads(bytes(arrays.pop("__meta__")).decode("utf-8"))
    return _decode_estimator(meta, arrays)


def iter_batches(n: int, batch: int) -> Iterable[slice]:
    for s in range(0, n, batch):
        yield slice(s, min(n, s + batch))

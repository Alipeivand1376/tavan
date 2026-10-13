"""Component registry and plugin discovery.

Every model, transform and constraint in ``tavan`` is registered under a
``kind`` (e.g. ``"generator"``) and a short ``name`` (e.g. ``"diffusion"``).
High-level APIs accept these names, and third-party packages can contribute
new components without modifying ``tavan``:

.. code-block:: python

    from tavan.registry import register
    from tavan.generate import BaseGenerator

    @register("generator", "my_model")
    class MyModel(BaseGenerator):
        ...

External packages may also expose an entry point in the ``tavan.plugins``
group; it is imported lazily the first time the registry is queried.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Type

_REGISTRY: Dict[str, Dict[str, type]] = {}
_PLUGINS_LOADED = False


def register(kind: str, name: str, aliases: Sequence[str] = ()) -> Callable[[type], type]:
    """Class decorator registering a component under ``kind``/``name``."""

    def deco(cls: type) -> type:
        table = _REGISTRY.setdefault(kind, {})
        for key in (name, *aliases):
            key = key.lower()
            if key in table and table[key] is not cls:
                raise KeyError(f"{kind} {key!r} is already registered to {table[key].__name__}")
            table[key] = cls
        cls._registry_kind = kind
        cls._registry_name = name.lower()
        return cls

    return deco


def _load_plugins() -> None:
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover
        return
    try:
        eps = entry_points()
        group = eps.select(group="tavan.plugins") if hasattr(eps, "select") else eps.get("tavan.plugins", [])
    except Exception:  # pragma: no cover - defensive
        return
    for ep in group:
        try:
            ep.load()
        except Exception as exc:  # pragma: no cover - plugin errors must not break tavan
            import warnings

            warnings.warn(f"failed to load tavan plugin {ep.name!r}: {exc}", stacklevel=2)


def get(kind: str, name: str) -> type:
    """Return the class registered as ``kind``/``name``."""
    _ensure_builtin()
    table = _REGISTRY.get(kind, {})
    key = name.lower()
    if key not in table:
        _load_plugins()
        table = _REGISTRY.get(kind, {})
    if key not in table:
        raise KeyError(f"unknown {kind} {name!r}; available: {available(kind)}")
    return table[key]


def available(kind: Optional[str] = None) -> List[str] | Dict[str, List[str]]:
    """List registered names (primary names only)."""
    _ensure_builtin()
    _load_plugins()

    def primary(k: str) -> List[str]:
        return sorted({c._registry_name for c in _REGISTRY.get(k, {}).values()})

    if kind is None:
        return {k: primary(k) for k in sorted(_REGISTRY)}
    return primary(kind)


def create(kind: str, name: str, **kwargs):
    """Instantiate a registered component."""
    return get(kind, name)(**kwargs)


def _ensure_builtin() -> None:
    # Import sub-packages so that their decorators run.
    from . import forecast, generate, preprocess, scenarios  # noqa: F401


def key_of(obj_or_cls) -> str:
    cls = obj_or_cls if isinstance(obj_or_cls, type) else type(obj_or_cls)
    kind = getattr(cls, "_registry_kind", None)
    name = getattr(cls, "_registry_name", None)
    if kind is None:
        raise TypeError(f"{cls.__name__} is not registered; decorate it with tavan.registry.register")
    return f"{kind}:{name}"


def from_key(key: str) -> Type:
    kind, name = key.split(":", 1)
    return get(kind, name)

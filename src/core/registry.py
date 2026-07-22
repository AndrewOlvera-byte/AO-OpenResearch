from __future__ import annotations
from typing import Callable, Dict, Iterable
from collections import defaultdict

_REGISTRY: Dict[str, Dict[str, Callable]] = defaultdict(dict)

def register(kind: str, name: str):
    def decorator(fn: Callable):
        existing = _REGISTRY[kind].get(name)
        if existing is not None and existing is not fn:
            same_symbol = (
                getattr(existing, "__module__", None) == getattr(fn, "__module__", None)
                and getattr(existing, "__qualname__", None)
                == getattr(fn, "__qualname__", None)
            )
            if not same_symbol:
                raise ValueError(
                    f"duplicate registry entry {kind!r}/{name!r}: "
                    f"{existing!r} is already registered"
                )
        _REGISTRY[kind][name] = fn
        return fn
    return decorator

def build(kind: str, **kwargs):
    if "type" not in kwargs:
        raise ValueError(f"building registry kind {kind!r} requires a 'type' field")
    component_type = kwargs.pop("type")
    factory = _REGISTRY.get(kind, {}).get(component_type)
    if factory is None:
        available = ", ".join(sorted(_REGISTRY.get(kind, {}))) or "<none>"
        raise KeyError(
            f"unknown {kind} type {component_type!r}; registered types: {available}"
        )
    return factory(**kwargs)


def registered(kind: str) -> Iterable[str]:
    """Return registered names for validation and user-facing diagnostics."""
    return tuple(sorted(_REGISTRY.get(kind, {})))

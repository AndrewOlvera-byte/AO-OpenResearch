from __future__ import annotations
from typing import Callable, Dict
from collections import defaultdict

_REGISTRY: Dict[str, Dict[str, Callable]] = defaultdict(dict)

def register(kind: str, name: str):
    def decorator(fn: Callable):
        _REGISTRY[kind][name] = fn
        return fn
    return decorator

def build(kind: str, **kwargs):
    t = kwargs.pop("type")
    return _REGISTRY[kind][t](**kwargs)


from __future__ import annotations
from typing import Callable, Dict, Optional, Any
from dataclasses import dataclass
from .types import Dataset, MetricResult

_REGISTRY: Dict[str, Callable[..., MetricResult]] = {}

def dq_metric(id: str):
    """Decorator to register a metric computation function.
    The wrapped function should return a MetricResult. Signature is free-form.
    """
    def deco(fn: Callable[..., MetricResult]):
        if not isinstance(id, str) or not id:
            raise ValueError("Metric id must be a non-empty string")
        if id in _REGISTRY:
            raise ValueError(f"Metric id already registered: {id}")
        _REGISTRY[id] = fn
        fn.__dq_metric_id__ = id  # type: ignore[attr-defined]
        return fn
    return deco

def list_metrics(prefix: Optional[str] = None):
    keys = sorted(_REGISTRY.keys())
    if prefix:
        keys = [k for k in keys if k.startswith(prefix)]
    return keys

def compute(id: str, /, *args, **kwargs) -> MetricResult:
    if id not in _REGISTRY:
        raise KeyError(f"Unknown metric id: {id}")
    return _REGISTRY[id](*args, **kwargs)

def clear_registry():
    """For testing only."""
    _REGISTRY.clear()

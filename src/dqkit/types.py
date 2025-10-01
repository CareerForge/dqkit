
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd
import hashlib
import json

@dataclass
class Dataset:
    df: pd.DataFrame
    name: str = "dataset"
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(str(list(self.df.columns)).encode())
        h.update(str(len(self.df)).encode())
        return h.hexdigest()[:16]

@dataclass
class MetricResult:
    id: str
    level: str  # "dataset"|"column"|"row"|"segment"
    target: Any
    value: Any
    unit: Optional[str] = None
    interpretation: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"

@dataclass
class RunReport:
    metrics: List[MetricResult]
    artifacts: Dict[str, str] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps([m.__dict__ for m in self.metrics], default=str)

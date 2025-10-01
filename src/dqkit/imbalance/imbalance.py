
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
from ..types import Dataset, MetricResult, RunReport

def _effective_num(counts: np.ndarray, beta: float = 0.999) -> float:
    """Cui et al. (Class-Balanced Loss). Larger when distribution is flatter."""
    counts = counts.astype(float)
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    return float(eff.sum())

def _gini(counts: np.ndarray) -> float:
    # Gini on class probability distribution
    p = counts / counts.sum() if counts.sum() > 0 else counts
    return float(1.0 - np.sum(p**2))

def _rarity_index(counts: np.ndarray) -> Dict[Any, float]:
    # rarity = 1 - normalized frequency
    total = counts.sum()
    if total == 0:
        return {}
    p = counts / total
    # normalize to [0,1] by (p_max - p_c) / (p_max - p_min) if denom>0 else zeros
    pmax, pmin = float(p.max()), float(p.min())
    denom = pmax - pmin if pmax > pmin else 1.0
    return {i: float((pmax - pc) / denom) for i, pc in enumerate(p)}

def measure_imbalance(ds: Dataset, y: str, beta: float = 0.999) -> RunReport:
    """Compute imbalance metrics for a label column.
    Metrics:
      - dq.imbalance.counts
      - dq.imbalance.ir (majority/minority)
      - dq.imbalance.effective_n (Cui beta)
      - dq.imbalance.gini (1 - sum p^2)  # higher means more even
      - dq.imbalance.rarity.<class_value> in [0,1], higher = rarer
    """
    ser = ds.df[y]
    counts = ser.value_counts(dropna=False)
    classes = counts.index.tolist()
    cnt_values = counts.values.astype(float)
    majority = float(cnt_values.max()) if len(cnt_values) else float('nan')
    minority = float(cnt_values.min()) if len(cnt_values) else float('nan')
    ir = float(majority / minority) if len(cnt_values) and minority > 0 else float('inf')

    metrics: List[MetricResult] = []
    metrics.append(MetricResult("dq.imbalance.counts", "dataset", y, {str(k): int(v) for k, v in counts.items()}))
    metrics.append(MetricResult("dq.imbalance.ir", "dataset", y, ir))
    metrics.append(MetricResult("dq.imbalance.effective_n", "dataset", y, _effective_num(cnt_values, beta=beta), meta={"beta": beta}))
    metrics.append(MetricResult("dq.imbalance.gini", "dataset", y, _gini(cnt_values)))

    rarity = _rarity_index(cnt_values)
    for idx, cls in enumerate(classes):
        metrics.append(MetricResult(f"dq.imbalance.rarity.{cls}", "dataset", cls, rarity.get(idx, 0.0)))

    return RunReport(metrics=metrics, meta={"dataset": ds.name, "label": y})

def simulate_rebalance(counts: Dict[Any, int], target: Union[str, Dict[Any, int]] = "uniform") -> Dict[str, Any]:
    """Given existing class counts, simulate target counts and return sampling plan deltas.
    target:
      - "uniform": raise all classes to max(counts)
      - dict: explicit desired counts per class
    Returns: { 'current': {...}, 'target': {...}, 'deltas': {...}, 'total_added': int }
    """
    cur = {str(k): int(v) for k, v in counts.items()}
    if target == "uniform":
        m = max(cur.values()) if cur else 0
        tgt = {k: m for k in cur.keys()}
    elif isinstance(target, dict):
        tgt = {str(k): int(v) for k, v in target.items()}
    else:
        raise ValueError("Unsupported target")
    # align keys
    all_keys = sorted(set(cur) | set(tgt))
    cur = {k: cur.get(k, 0) for k in all_keys}
    tgt = {k: tgt.get(k, 0) for k in all_keys}
    deltas = {k: tgt[k] - cur[k] for k in all_keys}
    return {"current": cur, "target": tgt, "deltas": deltas, "total_added": int(sum(max(0, d) for d in deltas.values()))}

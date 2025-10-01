
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from ..types import Dataset, MetricResult, RunReport
from ..representativeness.compare import compare as _compare

def measure_drift(current: Dataset, reference: Dataset, features: Optional[Sequence[str]] = None, metrics: Sequence[str] = ("psi","ks"), bins: int = 10, artifacts_dir: Optional[str] = None) -> RunReport:
    """Batch drift between a *current* dataset and a *reference* snapshot.
    Under the hood uses representativeness.compare with the same metrics.
    Adds aggregate drift score as the mean PSI across features.
    """
    rep = _compare(current, reference, features=features, metrics=metrics, bins=bins, artifacts_dir=artifacts_dir)
    # rename aggregate to drift namespace (keep values)
    out_metrics: List[MetricResult] = []
    for m in rep.metrics:
        if m.id.startswith("dq.represent."):
            out_metrics.append(MetricResult(m.id.replace("dq.represent.", "dq.drift."), m.level, m.target, m.value, unit=m.unit, interpretation=m.interpretation, meta=m.meta, version=m.version))
        else:
            out_metrics.append(m)
    # For clarity, keep aggregate under drift id too
    psi_vals = [m.value for m in out_metrics if m.id.startswith("dq.drift.psi.") and m.level == "column"]
    if psi_vals:
        out_metrics.append(MetricResult("dq.drift.psi.aggregate", "dataset", features or "*", float(np.nanmean(psi_vals))))
    return RunReport(metrics=out_metrics, artifacts=rep.artifacts, meta={"current": current.name, "reference": reference.name})

def measure_drift_history(history: Sequence[Dataset], features: Optional[Sequence[str]] = None, metrics: Sequence[str] = ("psi",), bins: int = 10) -> RunReport:
    """Given an ordered sequence of snapshots, compute drift between consecutive pairs and a trend.
    Returns per-step aggregate PSI and per-feature PSI for the last step.
    """
    if len(history) < 2:
        return RunReport(metrics=[MetricResult("dq.drift.history.steps", "dataset", "*", 0)], meta={})
    metrics_out: List[MetricResult] = [MetricResult("dq.drift.history.steps", "dataset", "*", len(history)-1)]
    agg_vals = []
    for i in range(1, len(history)):
        a, b = history[i-1], history[i]
        rep = _compare(b, a, features=features, metrics=metrics, bins=bins)  # drift from a->b
        psi_vals = [m.value for m in rep.metrics if m.id.startswith("dq.represent.psi.")]
        step_agg = float(np.nanmean(psi_vals)) if psi_vals else float("nan")
        metrics_out.append(MetricResult(f"dq.drift.psi.aggregate.step_{i}", "dataset", f"{a.name}->{b.name}", step_agg))
        # If last step, also emit per-feature PSI with drift namespace
        if i == len(history)-1:
            for m in rep.metrics:
                if m.id.startswith("dq.represent.psi."):
                    col = m.id.split(".")[-1]
                    metrics_out.append(MetricResult(f"dq.drift.psi.{col}", "column", col, m.value))
        agg_vals.append(step_agg)
    # simple trend = last - first (where not nan)
    valid = [v for v in agg_vals if not (isinstance(v, float) and (np.isnan(v)))]
    if len(valid) >= 2:
        trend = float(valid[-1] - valid[0])
        metrics_out.append(MetricResult("dq.drift.psi.trend", "dataset", "*", trend))
    return RunReport(metrics=metrics_out, meta={"n_snapshots": len(history)})

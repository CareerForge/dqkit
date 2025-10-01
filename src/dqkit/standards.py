
from __future__ import annotations
from typing import Dict, Optional
from .types import MetricResult, RunReport

# ---- Default bands & thresholds ----
PSI_BANDS = {  # Population Stability Index
    "good": 0.10,      # < 0.10 small shift
    "warn": 0.25,      # < 0.25 moderate shift
    # >= 0.25 bad
}
KS_BANDS = {          # Kolmogorov–Smirnov (0..1)
    "good": 0.10,
    "warn": 0.20,
}
MAXCORR_WARN = 0.95   # |corr| >= 0.95 considered redundant
VIF_WARN = 10.0       # VIF > 10 is problematic
IR_WARN = 10.0        # Imbalance ratio > 10 flagged
MISSING_RATE_WARN = 0.20  # per-column missing > 20%

def _band(value: float, good_thr: float, warn_thr: float) -> str:
    if value < good_thr: return "good"
    if value < warn_thr: return "warn"
    return "bad"

def interpret(m: MetricResult) -> MetricResult:
    """Attach an interpretation label (good/warn/bad) to known metric IDs.
    Leaves m.interpretation as-is for unknown metrics or non-numeric values.
    """
    try:
        v = float(m.value)
    except Exception:
        return m

    mid = m.id

    if mid.startswith("dq.represent.psi.") or mid.startswith("dq.drift.psi."):
        m.interpretation = _band(v, PSI_BANDS["good"], PSI_BANDS["warn"])
        return m

    if mid.startswith("dq.represent.ks.") or mid.startswith("dq.drift.ks."):
        m.interpretation = _band(v, KS_BANDS["good"], KS_BANDS["warn"])
        return m

    if mid.startswith("dq.redundancy.features.maxcorr."):
        m.interpretation = "bad" if v >= MAXCORR_WARN else "good"
        return m

    if mid.startswith("dq.redundancy.features.vif."):
        m.interpretation = "bad" if v > VIF_WARN else "good"
        return m

    if mid == "dq.imbalance.ir":
        m.interpretation = "bad" if v > IR_WARN else "good"
        return m

    if mid.startswith("dq.missing.rate."):
        m.interpretation = "bad" if v > MISSING_RATE_WARN else "good"
        return m

    # default: leave as-is
    return m

def apply_interpretations(report: RunReport) -> RunReport:
    """Return a new RunReport with interpretations applied to all metrics."""
    new_metrics = []
    for m in report.metrics:
        # copy to avoid mutating original
        mm = MetricResult(**{**m.__dict__})
        new_metrics.append(interpret(mm))
    return RunReport(metrics=new_metrics, artifacts=report.artifacts, meta=report.meta)

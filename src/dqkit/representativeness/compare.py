
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import os
import numpy as np
import pandas as pd
from ..types import Dataset, MetricResult, RunReport

def _common_numeric_bins(a: pd.Series, b: pd.Series, bins: int = 10) -> np.ndarray:
    both = pd.concat([a.dropna(), b.dropna()], axis=0)
    if both.empty:
        return np.array([0.0, 1.0])
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(both, qs))
    if len(edges) < 2:
        # fallback to min/max +- tiny epsilon
        lo = float(both.min())
        hi = float(both.max())
        if lo == hi:
            hi = lo + 1e-6
        edges = np.array([lo, hi])
    return edges

def _hist_probs(x: pd.Series, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(x.dropna().astype(float), bins=edges)
    counts = counts.astype(float)
    total = counts.sum()
    if total <= 0:
        return np.zeros_like(counts, dtype=float)
    return counts / total

def _psi_from_probs(pa: np.ndarray, pb: np.ndarray) -> float:
    # add epsilon to avoid log(0)
    eps = 1e-6
    pa = np.where(pa == 0, eps, pa)
    pb = np.where(pb == 0, eps, pb)
    return float(((pa - pb) * np.log(pa / pb)).sum())

def _psi_numeric(a: pd.Series, b: pd.Series, bins: int = 10) -> Tuple[float, np.ndarray]:
    edges = _common_numeric_bins(a, b, bins=bins)
    pa = _hist_probs(a, edges)
    pb = _hist_probs(b, edges)
    return _psi_from_probs(pa, pb), edges

def _psi_categorical(a: pd.Series, b: pd.Series) -> float:
    cats = sorted(set(a.dropna().unique().tolist()) | set(b.dropna().unique().tolist()))
    if not cats:
        return float("nan")
    va = a.value_counts(normalize=True)
    vb = b.value_counts(normalize=True)
    pa = np.array([va.get(c, 0.0) for c in cats], dtype=float)
    pb = np.array([vb.get(c, 0.0) for c in cats], dtype=float)
    return _psi_from_probs(pa, pb)

def _ks_numeric(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna().astype(float); b = b.dropna().astype(float)
    if a.empty or b.empty:
        return float("nan")
    # empirical CDF grid over combined values
    grid = np.sort(np.unique(np.concatenate([a.values, b.values])))
    Fa = np.searchsorted(np.sort(a.values), grid, side="right") / len(a)
    Fb = np.searchsorted(np.sort(b.values), grid, side="right") / len(b)
    return float(np.max(np.abs(Fa - Fb)))

def compare(a: Dataset, b: Dataset, features: Optional[Sequence[str]] = None, metrics: Sequence[str] = ("psi","ks"), bins: int = 10, artifacts_dir: Optional[str] = None) -> RunReport:
    """Compare candidate dataset `a` vs reference dataset `b` feature-wise.
    Metrics supported:
      - 'psi' (numeric via quantile bins; categorical via frequency)
      - 'ks'  (numeric Kolmogorov-Smirnov distance)
    Returns per-feature MetricResults and optional artifacts (bin edges per numeric column).
    """
    df_a, df_b = a.df, b.df
    feats = list(features) if features is not None else [c for c in df_a.columns if c in df_b.columns]
    out: List[MetricResult] = []
    artifacts: Dict[str, str] = {}

    for col in feats:
        sa, sb = df_a[col], df_b[col]
        is_num = pd.api.types.is_numeric_dtype(sa) and pd.api.types.is_numeric_dtype(sb)
        if "psi" in metrics:
            if is_num:
                val, edges = _psi_numeric(sa, sb, bins=bins)
                out.append(MetricResult(f"dq.represent.psi.{col}", "column", col, float(val), meta={"bins": int(bins)}))
                if artifacts_dir is not None:
                    os.makedirs(artifacts_dir, exist_ok=True)
                    path = os.path.join(artifacts_dir, f"{col}_bins.csv")
                    pd.DataFrame({"edge": edges}).to_csv(path, index=False)
                    artifacts[f"artifact.represent.{col}.bins"] = path
            else:
                val = _psi_categorical(sa, sb)
                out.append(MetricResult(f"dq.represent.psi.{col}", "column", col, float(val)))
        if "ks" in metrics and is_num:
            ks = _ks_numeric(sa, sb)
            out.append(MetricResult(f"dq.represent.ks.{col}", "column", col, float(ks)))

    # aggregate (average over numeric PSI only if present)
    psi_vals = [m.value for m in out if m.id.startswith("dq.represent.psi.")]
    if psi_vals:
        out.append(MetricResult("dq.represent.psi.aggregate", "dataset", feats, float(np.nanmean(psi_vals))))
    return RunReport(metrics=out, artifacts=artifacts, meta={"left": a.name, "right": b.name})

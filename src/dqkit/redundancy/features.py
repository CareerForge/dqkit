
from __future__ import annotations
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
from ..types import Dataset, MetricResult, RunReport

def _vif(X: np.ndarray) -> np.ndarray:
    # Basic VIF: regress each column on others via closed form (X'X)^{-1}
    # Add small ridge for stability.
    n, p = X.shape
    vifs = np.zeros(p)
    for j in range(p):
        y = X[:, j]
        Z = np.delete(X, j, axis=1)
        # add intercept
        Z1 = np.c_[np.ones(n), Z]
        XtX = Z1.T @ Z1 + 1e-6 * np.eye(Z1.shape[1])
        beta = np.linalg.solve(XtX, Z1.T @ y)
        y_hat = Z1 @ beta
        rss = np.sum((y - y_hat)**2)
        tss = np.sum((y - y.mean())**2) + 1e-12
        r2 = 1.0 - rss / tss
        vifs[j] = 1.0 / max(1.0 - r2, 1e-8)
    return vifs

def measure_feature_redundancy(ds: Dataset, columns: Optional[Sequence[str]] = None, corr_method: str = "pearson", vif: bool = True, mi: bool = False, artifacts_dir: Optional[str] = None) -> RunReport:
    """Compute feature redundancy diagnostics:
      - correlation matrix (Pearson/Spearman) -> artifact CSV
      - VIF per numeric feature
      - simple redundancy score = max_{k!=j} |corr_{jk}| per feature
    """
    df = ds.df if columns is None else ds.df[list(columns)]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    metrics: List[MetricResult] = []
    artifacts: Dict[str, str] = {}

    if num_cols:
        corr = df[num_cols].corr(method=corr_method)
        # redundancy score = max abs corr with others (fill 0 on diagonal)
        abs_corr = corr.abs().copy()
        np.fill_diagonal(abs_corr.values, 0.0)
        for c in num_cols:
            score = float(abs_corr.loc[c].max())
            metrics.append(MetricResult(f"dq.redundancy.features.maxcorr.{c}", "column", c, score))
        if artifacts_dir is not None:
            import os
            os.makedirs(artifacts_dir, exist_ok=True)
            path = os.path.join(artifacts_dir, f"{ds.name}__corr_{corr_method}.csv")
            corr.to_csv(path)
            artifacts[f"artifact.redundancy.corr_{corr_method}"] = path

        if vif and len(num_cols) >= 2:
            X = df[num_cols].to_numpy(float)
            # center & scale to unit var to stabilize
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
            v = _vif(X)
            for c, vv in zip(num_cols, v):
                metrics.append(MetricResult(f"dq.redundancy.features.vif.{c}", "column", c, float(vv)))

    # Aggregate overall redundancy index (max of maxcorr)
    if num_cols:
        agg = max([m.value for m in metrics if m.id.startswith("dq.redundancy.features.maxcorr.")], default=0.0)
        metrics.append(MetricResult("dq.redundancy.features.aggregate", "dataset", num_cols, float(agg)))

    return RunReport(metrics=metrics, artifacts=artifacts, meta={"dataset": ds.name, "n_numeric": len(num_cols)})

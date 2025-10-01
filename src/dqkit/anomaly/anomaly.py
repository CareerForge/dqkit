
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import os
import numpy as np
import pandas as pd
from ..types import Dataset, MetricResult, RunReport

try:
    from sklearn.ensemble import IsolationForest
except Exception:  # pragma: no cover
    IsolationForest = None

def _robust_z(x: np.ndarray) -> np.ndarray:
    # robust z = |x - median| / (1.4826 * MAD)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    denom = 1.4826 * mad if mad > 0 else (np.nanstd(x) + 1e-12)
    return np.abs(x - med) / denom

def _iqr_score(x: np.ndarray) -> np.ndarray:
    q1 = np.nanpercentile(x, 25)
    q3 = np.nanpercentile(x, 75)
    iqr = q3 - q1
    if iqr <= 0:
        return np.zeros_like(x, dtype=float)
    # 1.5*IQR rule → score grows linearly beyond fences
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    s = np.zeros_like(x, dtype=float)
    s[x < lower] = (lower - x[x < lower]) / (iqr + 1e-12)
    s[x > upper] = (x[x > upper] - upper) / (iqr + 1e-12)
    return s

def score_outliers(ds: Dataset, columns: Optional[Sequence[str]] = None, method: str = "auto", contamination: float = 0.01, artifacts_dir: Optional[str] = None) -> RunReport:
    """Score per-row anomalies.
    - method="auto": combine per-column robust z and IQR into a 0..1 score via sigmoid; max across columns.
    - method="iforest": IsolationForest multivariate score (requires scikit-learn).
    Outputs:
      - dq.anomaly.rate  (fraction of rows flagged given default threshold from contamination)
      - dq.anomaly.threshold  (score threshold used)
      - dq.anomaly.score.<col> (avg column-level outlierness for numeric columns, auto mode)
    Artifact: CSV of rows with score >= threshold.
    """
    df = ds.df if columns is None else ds.df[list(columns)]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    n = len(df)
    if n == 0 or not num_cols:
        return RunReport(metrics=[MetricResult("dq.anomaly.rate", "dataset", "*", 0.0)], meta={"dataset": ds.name, "n": n})

    X = df[num_cols].to_numpy(float)
    # simple mean impute
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    if method == "iforest":
        if IsolationForest is None:
            raise ImportError("scikit-learn required for IsolationForest method")
        iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
        iso.fit(X)
        # higher scores indicate more normal; convert to anomaly score 0..1
        s = -iso.score_samples(X)
        # normalize to 0..1
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    else:
        # auto: combine robust z and iqr per column
        z_scores = np.column_stack([_robust_z(X[:, j]) for j in range(X.shape[1])])
        iqr_scores = np.column_stack([_iqr_score(X[:, j]) for j in range(X.shape[1])])
        col_scores = np.maximum(z_scores, iqr_scores)
        # squash with sigmoid to [0,1], using 3 as a typical robust z cutoff
        s = 1.0 / (1.0 + np.exp(-(col_scores - 3.0)))
        # take max across columns for per-row score
        s = s.max(axis=1)

    # threshold by quantile from contamination
    thr = float(np.quantile(s, 1.0 - contamination)) if n > 1 else 1.0
    flagged = s >= thr
    rate = float(flagged.mean())

    metrics: List[MetricResult] = [
        MetricResult("dq.anomaly.rate","dataset","*", rate),
        MetricResult("dq.anomaly.threshold","dataset","*", thr),
    ]
    if method != "iforest":
        # emit column averages to help locate problem columns
        for j, c in enumerate(num_cols):
            metrics.append(MetricResult(f"dq.anomaly.score.{c}", "column", c, float(np.mean(1.0 / (1.0 + np.exp(-(np.maximum(_robust_z(X[:, j]), _iqr_score(X[:, j])) - 3.0)))))))

    artifacts: Dict[str, str] = {}
    if artifacts_dir is not None and flagged.any():
        os.makedirs(artifacts_dir, exist_ok=True)
        out = df.copy()
        out["dq_anomaly_score"] = s
        out = out.loc[flagged]
        path = os.path.join(artifacts_dir, f"{ds.name}__anomalies.csv")
        out.to_csv(path, index=False)
        artifacts["artifact.anomaly.rows"] = path

    return RunReport(metrics=metrics, artifacts=artifacts, meta={"dataset": ds.name, "method": method, "columns": num_cols})

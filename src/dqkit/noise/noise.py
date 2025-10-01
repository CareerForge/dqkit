
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple, Any
import os
import numpy as np
import pandas as pd
from ..types import Dataset, MetricResult, RunReport

def estimate_label_noise(
    ds: Dataset,
    y: str,
    proba: Optional[np.ndarray] = None,
    classes: Optional[Sequence[Any]] = None,
    features: Optional[Sequence[str]] = None,
    threshold: Optional[float] = None,
    artifacts_dir: Optional[str] = None,
) -> RunReport:
    """
    Estimate label noise for a classification label column.

    Two modes:
      1) Probabilistic (preferred): provide `proba` as an (n_samples, n_classes) numpy array aligned to ds.df.
         - Uses confidence in the observed label: suspicion_i = 1 - proba[i, y_i_index].
      2) Heuristic (no proba): 1-NN agreement over numeric `features`.
         - suspicion_i = 1 if nearest neighbor has a different label, else 0.

    Returns MetricResults:
      - dq.noise.rate.overall
      - dq.noise.rate.class.<class_value>
      - dq.noise.suspect_count (number of rows with suspicion > threshold)
    Also writes artifact CSV of suspected rows if artifacts_dir is provided.
    """
    df = ds.df
    y_series = df[y]
    if classes is None:
        classes = sorted(y_series.dropna().unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    n = len(df)
    susp: np.ndarray

    if proba is not None:
        proba = np.asarray(proba)
        if proba.shape[0] != n or proba.shape[1] != len(classes):
            raise ValueError("proba shape must be (n_samples, n_classes) with classes provided or inferred.")
        # map observed label to index; if label not in classes, suspicious=1.0
        idxs = y_series.map(lambda v: class_to_idx.get(v, None))
        susp = np.ones(n, dtype=float)
        mask_known = idxs.notna().to_numpy()
        susp[mask_known] = 1.0 - proba[np.arange(n)[mask_known], idxs[mask_known].astype(int)]
        default_threshold = 0.8
    else:
        # 1-NN heuristic over numeric features
        if features is None:
            features = [c for c in df.columns if c != y and pd.api.types.is_numeric_dtype(df[c])]
        X = df[features].to_numpy(dtype=float)
        # handle missing by imputing column means
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
        # compute pairwise distances
        # for large n this is O(n^2) and intended for small datasets/tests
        dists = _pairwise_squared_euclidean(X)
        np.fill_diagonal(dists, np.inf)
        nn_idx = dists.argmin(axis=1)
        nn_label = y_series.to_numpy()[nn_idx]
        susp = (nn_label != y_series.to_numpy()).astype(float)
        default_threshold = 1.0

    # aggregate metrics
    overall = float(np.nanmean(susp)) if len(susp) else float("nan")
    metrics: List[MetricResult] = [
        MetricResult("dq.noise.rate.overall", "dataset", y, overall)
    ]
    for c in classes:
        cls_mask = (y_series == c).to_numpy()
        if cls_mask.any():
            metrics.append(MetricResult(f"dq.noise.rate.class.{c}", "dataset", c, float(np.nanmean(susp[cls_mask]))))

    # suspected rows
    thr = default_threshold if threshold is None else float(threshold)
    suspects_mask = susp > thr
    suspect_count = int(suspects_mask.sum())
    metrics.append(MetricResult("dq.noise.suspect_count", "dataset", y, suspect_count))
    artifacts: Dict[str, str] = {}
    if artifacts_dir is not None and suspect_count > 0:
        os.makedirs(artifacts_dir, exist_ok=True)
        out = df.loc[suspects_mask].copy()
        out["suspect_score"] = susp[suspects_mask]
        path = os.path.join(artifacts_dir, f"{ds.name}__suspected_label_errors.csv")
        out.to_csv(path, index=False)
        artifacts["artifact.noise.suspects"] = path

    return RunReport(metrics=metrics, artifacts=artifacts, meta={"dataset": ds.name, "mode": "proba" if proba is not None else "1nn"})

def _pairwise_squared_euclidean(X: np.ndarray) -> np.ndarray:
    # (x - y)^2 = x^2 + y^2 - 2xy
    sq = np.sum(X*X, axis=1, keepdims=True)
    d2 = sq + sq.T - 2 * (X @ X.T)
    # numerical noise can make tiny negatives; clip
    np.maximum(d2, 0.0, out=d2)
    return d2

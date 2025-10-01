
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import os
import numpy as np
import pandas as pd
from ..types import Dataset, MetricResult, RunReport

def find_duplicates(ds: Dataset, keys: Optional[Sequence[str]] = None, artifacts_dir: Optional[str]=None) -> RunReport:
    """Compute exact duplicate rate over all columns or a subset of keys."""
    df = ds.df if keys is None else ds.df[list(keys)]
    dup_mask = df.duplicated(keep=False)
    rate = float(dup_mask.mean())
    metrics = [MetricResult("dq.redundancy.rows.duplicate_rate", "dataset", keys or "*", rate)]
    artifacts: Dict[str, str] = {}
    if artifacts_dir is not None and rate > 0.0:
        os.makedirs(artifacts_dir, exist_ok=True)
        path = os.path.join(artifacts_dir, f"{ds.name}__duplicate_groups.csv")
        out = ds.df.loc[dup_mask].copy()
        out.to_csv(path, index=False)
        artifacts["artifact.redundancy.duplicates"] = path
    return RunReport(metrics=metrics, artifacts=artifacts, meta={"dataset": ds.name})

def _cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    # Normalize rows
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    return Xn @ Xn.T

def _jaccard_sim_strings(col: pd.Series) -> np.ndarray:
    # tokenize by whitespace; return pairwise similarities (n x n) — O(n^2), for small sets/tests
    toks = [set(str(v).split()) if pd.notna(v) else set() for v in col.tolist()]
    n = len(toks)
    S = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            a, b = toks[i], toks[j]
            num = len(a & b)
            den = len(a | b) if (a or b) else 1
            sim = num/den
            S[i,j] = S[j,i] = sim
    return S

def find_near_duplicates(
    ds: Dataset,
    numeric_cols: Optional[Sequence[str]] = None,
    text_cols: Optional[Sequence[str]] = None,
    threshold: float = 0.95,
    artifacts_dir: Optional[str]=None
) -> RunReport:
    """Identify near-duplicate row pairs via cosine similarity on numeric columns and Jaccard on text columns.
    A pair is considered near-duplicate if either numeric cosine >= threshold or (if numeric_cols not provided),
    text Jaccard >= threshold. Returns rate = (#pairs above threshold) / (n choose 2).
    Artifacts: CSV with pairs (i, j, sim, channel).
    """
    df = ds.df
    n = len(df)
    if n < 2:
        return RunReport(metrics=[MetricResult("dq.redundancy.rows.near_dup_rate", "dataset", "*", 0.0)], meta={"dataset": ds.name})

    pairs = []
    total_pairs = n*(n-1)//2

    if numeric_cols:
        X = df[list(numeric_cols)].to_numpy(float)
        # impute NaNs with column means
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
        S = _cosine_similarity_matrix(X)
        ch = "numeric"
        for i in range(n):
            for j in range(i+1, n):
                if S[i,j] >= threshold:
                    pairs.append((i,j, float(S[i,j]), ch))

    if text_cols:
        for c in text_cols:
            S = _jaccard_sim_strings(df[c])
            ch = f"text:{c}"
            for i in range(n):
                for j in range(i+1, n):
                    if S[i,j] >= threshold:
                        pairs.append((i,j, float(S[i,j]), ch))

    # deduplicate pairs (keep max sim across channels)
    pairmap: Dict[Tuple[int,int], Tuple[float,str]] = {}
    for i,j,sim,ch in pairs:
        key = (i,j)
        if key not in pairmap or sim > pairmap[key][0]:
            pairmap[key] = (sim, ch)

    near_dup_count = len(pairmap)
    rate = near_dup_count / total_pairs if total_pairs > 0 else 0.0

    metrics = [MetricResult("dq.redundancy.rows.near_dup_rate", "dataset", "*", float(rate), meta={"threshold": threshold})]
    artifacts: Dict[str, str] = {}
    if artifacts_dir is not None and near_dup_count > 0:
        os.makedirs(artifacts_dir, exist_ok=True)
        import pandas as pd
        out = pd.DataFrame([(i,j,sim,ch) for (i,j),(sim,ch) in pairmap.items()], columns=["i","j","similarity","channel"])
        path = os.path.join(artifacts_dir, f"{ds.name}__near_duplicates.csv")
        out.to_csv(path, index=False)
        artifacts["artifact.redundancy.near_duplicates"] = path

    return RunReport(metrics=metrics, artifacts=artifacts, meta={"dataset": ds.name, "n_pairs": total_pairs})

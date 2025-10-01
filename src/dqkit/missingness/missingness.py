
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import os
import itertools
import numpy as np
import pandas as pd
from ..types import Dataset, MetricResult, RunReport

def analyze_missingness(ds: Dataset, columns: Optional[Sequence[str]] = None, artifacts_dir: Optional[str] = None, top_k_patterns: int = 10) -> RunReport:
    """
    Compute missingness metrics for a dataset:
      - dataset row-level missingness rate
      - per-column missingness rate
      - pairwise co-occurrence (P(both missing))
      - top-k missingness patterns (bitmask over selected columns)
    Saves artifacts (CSV) for heatmap/co-occurrence and patterns if artifacts_dir is provided.
    """
    df = ds.df if columns is None else ds.df[list(columns)]
    metrics: List[MetricResult] = []
    artifacts: Dict[str, str] = {}

    miss = df.isna()
    row_rate = float(miss.any(axis=1).mean())
    metrics.append(MetricResult("dq.missing.row_rate", "dataset", "*", row_rate))

    # per-column
    for col in df.columns:
        col_rate = float(miss[col].mean())
        metrics.append(MetricResult(f"dq.missing.rate.{col}", "column", col, col_rate))

    # pairwise co-occurrence matrix
    cols = list(df.columns)
    if len(cols) >= 2:
        pair_rows = []
        for i, j in itertools.combinations(range(len(cols)), 2):
            ci, cj = cols[i], cols[j]
            p_both = float((miss[ci] & miss[cj]).mean())
            pair_rows.append((ci, cj, p_both))
            metrics.append(MetricResult(f"dq.missing.cooccur.{ci}.{cj}", "dataset", [ci, cj], p_both))
        if artifacts_dir is not None:
            os.makedirs(artifacts_dir, exist_ok=True)
            import pandas as pd
            co_df = pd.DataFrame(pair_rows, columns=["col_i","col_j","p_both_missing"])
            co_path = os.path.join(artifacts_dir, "missing_cooccurrence.csv")
            co_df.to_csv(co_path, index=False)
            artifacts["artifact.missing.cooccurrence"] = co_path

    # top-k patterns over a small set of columns (cap at 20 columns to avoid blowup)
    cap_cols = cols[:20]
    if len(cap_cols) > 0:
        bitcols = miss[cap_cols].astype(int)
        # build bitmask string key per row, e.g., "0101"
        keys = bitcols.apply(lambda r: "".join(map(str, r.values.tolist())), axis=1)
        vc = keys.value_counts().head(top_k_patterns)
        patterns = [{"pattern": k, "count": int(v)} for k, v in vc.items()]
        metrics.append(MetricResult("dq.missing.top_patterns", "dataset", cap_cols, patterns))
        if artifacts_dir is not None:
            pat_path = os.path.join(artifacts_dir, "missing_top_patterns.csv")
            vc.to_frame("count").to_csv(pat_path)
            artifacts["artifact.missing.top_patterns"] = pat_path

    return RunReport(metrics=metrics, artifacts=artifacts, meta={"dataset": ds.name})

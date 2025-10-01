
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence
import math
import numpy as np
import pandas as pd
from ..types import Dataset, MetricResult, RunReport

_NUM_QS = [0.05, 0.25, 0.5, 0.75, 0.95]

def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _is_datetime(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)

def _entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return float('nan')
    p = counts.astype(float) / float(total)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def profile(ds: Dataset, columns: Optional[Sequence[str]] = None, bins: int = 10, artifacts_dir: Optional[str] = None) -> RunReport:
    """
    Compute basic per-column profiling metrics.
    - count (non-null), missing_rate, distinct, dtype
    - numeric: min, max, mean, std, quantiles, histogram (artifact)
    - categorical (object/string): top-k frequencies, entropy
    - datetime: min, max
    Returns a RunReport with MetricResult entries; artifacts (CSV) saved if artifacts_dir provided.
    """
    df = ds.df if columns is None else ds.df[list(columns)]
    metrics: List[MetricResult] = []
    artifacts: Dict[str, str] = {}

    for col in df.columns:
        s = df[col]
        non_null = int(s.notna().sum())
        missing_rate = float(s.isna().mean())
        distinct = int(s.nunique(dropna=True))
        metrics.append(MetricResult(f"dq.profile.count.{col}", "column", col, non_null))
        metrics.append(MetricResult(f"dq.profile.missing_rate.{col}", "column", col, missing_rate))
        metrics.append(MetricResult(f"dq.profile.distinct.{col}", "column", col, distinct))
        metrics.append(MetricResult(f"dq.profile.dtype.{col}", "column", col, str(s.dtype)))

        if _is_numeric(s):
            s_num = s.dropna().astype(float)
            if len(s_num) > 0:
                metrics.extend([
                    MetricResult(f"dq.profile.min.{col}", "column", col, float(s_num.min())),
                    MetricResult(f"dq.profile.max.{col}", "column", col, float(s_num.max())),
                    MetricResult(f"dq.profile.mean.{col}", "column", col, float(s_num.mean())),
                    MetricResult(f"dq.profile.std.{col}", "column", col, float(s_num.std(ddof=1)) if len(s_num)>1 else 0.0),
                ])
                # quantiles
                qs = np.quantile(s_num, _NUM_QS)
                for q, v in zip(_NUM_QS, qs):
                    metrics.append(MetricResult(f"dq.profile.q{int(q*100)}.{col}", "column", col, float(v)))
                # histogram
                hist_counts, edges = np.histogram(s_num, bins=bins)
                if artifacts_dir is not None:
                    import os
                    os.makedirs(artifacts_dir, exist_ok=True)
                    import pandas as pd
                    hist_df = pd.DataFrame({
                        "left": edges[:-1],
                        "right": edges[1:],
                        "count": hist_counts
                    })
                    path = os.path.join(artifacts_dir, f"{col}_hist.csv")
                    hist_df.to_csv(path, index=False)
                    artifacts[f"artifact.hist.{col}"] = path

        elif _is_datetime(s):
            s_dt = s.dropna()
            if len(s_dt) > 0:
                metrics.append(MetricResult(f"dq.profile.min.{col}", "column", col, s_dt.min()))
                metrics.append(MetricResult(f"dq.profile.max.{col}", "column", col, s_dt.max()))

        else:
            # treat as categorical/text-like
            s_str = s.dropna().astype(str)
            if len(s_str) > 0:
                vc = s_str.value_counts()
                topk = vc.head(10).to_dict()
                metrics.append(MetricResult(f"dq.profile.topk.{col}", "column", col, topk))
                ent = _entropy_from_counts(vc.values.astype(int))
                metrics.append(MetricResult(f"dq.profile.entropy.{col}", "column", col, ent, unit="bits"))
                if artifacts_dir is not None:
                    import os
                    os.makedirs(artifacts_dir, exist_ok=True)
                    freq_path = os.path.join(artifacts_dir, f"{col}_freq.csv")
                    vc.to_frame("count").to_csv(freq_path)
                    artifacts[f"artifact.freq.{col}"] = freq_path

    # dataset-level summary
    metrics.append(MetricResult("dq.profile.n_rows", "dataset", "*", int(len(df))))
    metrics.append(MetricResult("dq.profile.n_cols", "dataset", "*", int(df.shape[1])))
    return RunReport(metrics=metrics, artifacts=artifacts, meta={"dataset": ds.name})

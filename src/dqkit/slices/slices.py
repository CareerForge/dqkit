
from __future__ import annotations
from typing import Callable, Dict, List, Optional
import pandas as pd
from ..types import Dataset, RunReport, MetricResult

def evaluate_by_segment(ds: Dataset, segments: Dict[str, pd.Series], metric_fns: List[Callable[[Dataset], RunReport]]) -> RunReport:
    """Compute metrics per segment by applying provided metric functions.
    - segments: mapping name -> boolean mask aligned to ds.df.index, or Series of categories
    - metric_fns: list of functions (Dataset->RunReport)
    Returns: combined RunReport with segment suffix in metric IDs and meta.
    """
    all_metrics: List[MetricResult] = []
    artifacts = {}
    for seg_name, mask in segments.items():
        if mask.dtype == bool:
            seg_df = ds.df[mask].copy()
            sub = Dataset(seg_df, name=f"{ds.name}[{seg_name}]")
            for fn in metric_fns:
                rep = fn(sub)
                for m in rep.metrics:
                    mm = MetricResult(
                        id=m.id + f".segment[{seg_name}]",
                        level=m.level,
                        target=m.target,
                        value=m.value,
                        unit=m.unit,
                        interpretation=m.interpretation,
                        meta={**m.meta, "segment": seg_name},
                        version=m.version
                    )
                    all_metrics.append(mm)
                artifacts.update({f"{k}.segment[{seg_name}]": v for k,v in rep.artifacts.items()})
        else:
            # categorical segments: group by category values
            for cat, idx in mask.groupby(mask):
                seg_df = ds.df.loc[idx.index]
                sub = Dataset(seg_df, name=f"{ds.name}[{seg_name}={cat}]")
                for fn in metric_fns:
                    rep = fn(sub)
                    for m in rep.metrics:
                        mm = MetricResult(
                            id=m.id + f".segment[{seg_name}={cat}]",
                            level=m.level,
                            target=m.target,
                            value=m.value,
                            unit=m.unit,
                            interpretation=m.interpretation,
                            meta={**m.meta, "segment": f"{seg_name}={cat}"},
                            version=m.version
                        )
                        all_metrics.append(mm)
                    artifacts.update({f"{k}.segment[{seg_name}={cat}]": v for k,v in rep.artifacts.items()})
    return RunReport(metrics=all_metrics, artifacts=artifacts, meta={"dataset": ds.name, "segments": list(segments.keys())})

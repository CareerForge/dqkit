
from __future__ import annotations
from typing import Dict, List, Optional, Iterable
import os, json, time, hashlib
from ..types import RunReport, MetricResult

def _run_id(report: RunReport) -> str:
    h = hashlib.sha256()
    for m in sorted(report.metrics, key=lambda x: x.id):
        h.update(m.id.encode())
        h.update(str(m.value).encode())
    return h.hexdigest()[:12]

def log_run(report: RunReport, store: str) -> str:
    """Persist a RunReport to a directory store as JSON lines.
    Returns the path to the saved file.
    """
    os.makedirs(store, exist_ok=True)
    rid = _run_id(report)
    ts = int(time.time())
    path = os.path.join(store, f"run_{ts}_{rid}.jsonl")
    with open(path, "w") as f:
        for m in report.metrics:
            f.write(json.dumps(m.__dict__, default=str) + "\n")
    # save meta
    with open(path + ".meta.json", "w") as f:
        json.dump(report.meta, f)
    return path

def load_history(store: str) -> List[RunReport]:
    """Load all runs from a store directory (sorted by time)."""
    runs = []
    if not os.path.isdir(store):
        return runs
    files = sorted([p for p in os.listdir(store) if p.endswith('.jsonl')])
    for p in files:
        metrics = []
        with open(os.path.join(store, p), 'r') as f:
            for line in f:
                d = json.loads(line)
                metrics.append(MetricResult(**d))
        meta_path = os.path.join(store, p + '.meta.json')
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as mf:
                meta = json.load(mf)
        runs.append(RunReport(metrics=metrics, meta=meta))
    return runs

def diff_metrics(a: RunReport, b: RunReport) -> Dict[str, float]:
    """Compute simple diffs for numeric metric values shared between two reports: b - a."""
    am = {m.id: m.value for m in a.metrics if isinstance(m.value, (int, float))}
    bm = {m.id: m.value for m in b.metrics if isinstance(m.value, (int, float))}
    keys = set(am) & set(bm)
    return {k: float(bm[k]) - float(am[k]) for k in sorted(keys)}

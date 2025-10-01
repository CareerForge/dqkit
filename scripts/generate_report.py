
import os
import pandas as pd
from dqkit.types import Dataset, RunReport
from dqkit.profiling import profile
from dqkit.missingness import analyze_missingness
from dqkit.imbalance import measure_imbalance
from dqkit.report import render

def main():
    # tiny demo dataset
    df = pd.DataFrame({
        "age": [20, 25, 30, None, 45, 50],
        "fare": [100.0, 80.5, 120.2, 99.9, 200.0, 55.0],
        "label": [0, 0, 1, 1, 1, 1],
    })
    ds = Dataset(df, name="demo")

    reps = []
    reps.append(profile(ds, artifacts_dir="artifacts"))
    reps.append(analyze_missingness(ds, artifacts_dir="artifacts"))
    reps.append(measure_imbalance(ds, y="label"))

    # Merge reports (simple concat of metrics & artifacts)
    metrics = []
    artifacts = {}
    for r in reps:
        metrics.extend(r.metrics)
        artifacts.update(r.artifacts)

    final = RunReport(metrics=metrics, artifacts=artifacts, meta={"example": "ci-build"})
    paths = render(final, out_dir="dq_reports", filename="report.html")
    print(paths["html"])

if __name__ == "__main__":
    main()

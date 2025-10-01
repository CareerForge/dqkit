
# Usage

## Dataset wrapper

```python
from dqkit.io.pandas_io import from_dataframe
ds = from_dataframe(df, name="mydata")
```

## Running metrics

- Validation: `validate(ds, spec)`
- Profiling: `profile(ds)`
- Missingness: `analyze_missingness(ds)`
- Noise: `estimate_label_noise(ds, y="label", proba=proba)`
- Imbalance: `measure_imbalance(ds, y="label")`
- Redundancy: `find_duplicates(ds)`, `measure_feature_redundancy(ds)`
- Representativeness: `compare(train_ds, test_ds)`
- Drift: `measure_drift(current, reference)`
- Anomaly: `score_outliers(ds)`
- Logging: `log_run(report, store="metrics/")`
- Checks: `run_checks(report, checks)`
- Reporting: `render(report)`
- Standards: `apply_interpretations(report)`
- Registry: `@dq_metric` to add custom metrics
- Slices: `evaluate_by_segment(ds, segments, [metric_fn])`

## Reports

```python
from dqkit.report import render
render(run_report, out_dir="dq_reports")
```

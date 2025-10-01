
# dqkit

**Data Quality Evaluation Toolkit** (pandas-first)

A Python Library for Testing and Ensuring Data Quality in Pandas.

High-quality data is critical for the reliability of modern machine learning (ML) and data-driven decision-making systems. Poor data quality can lead to biased outcomes, reduced model performance, and misleading insights. Despite its importance, data quality testing remains ad hoc and fragmented across tools and workflows. We present dqkit, an open-source Python library that provides a unified framework for validating, profiling, and monitoring the quality of tabular data, with a primary focus on pandas dataframes. The toolkit implements a wide range of data quality checks—such as validation against standards and thresholds, missingness analysis, imbalance detection, redundancy checks, representativeness evaluation, noise detection, drift monitoring, logging, and reporting—through a standardized, extensible interface. dqkit is designed to be easily integrated into existing machine learning pipelines, continuous integration workflows, and reporting systems. By combining breadth of coverage with lightweight usability, dqkit aims to establish itself as a community standard for reproducible data quality testing.

## Features

- Schema & validation rules
- Profiling (stats, histograms, entropy)
- Missingness patterns
- Label noise estimation
- Class imbalance measures
- Row/feature redundancy detection
- Representativeness (PSI, KS)
- Drift over time
- Anomaly/outlier scoring
- Logging & longitudinal tracking
- Checks-as-code for CI
- Report generation (HTML/Markdown)
- Standards & thresholds
- Metric registry for plugins
- Segments / fairness slices

## Quickstart

```bash
pip install dqkit
```

```python
import pandas as pd
from dqkit.io.pandas_io import from_dataframe
from dqkit.profiling import profile
from dqkit.missingness import analyze_missingness
from dqkit.imbalance import measure_imbalance
from dqkit.report import render
from dqkit.standards import apply_interpretations
from dqkit.types import RunReport

df = pd.DataFrame({"age":[20,25,None,40], "label":[0,0,1,1]})
ds = from_dataframe(df, name="toy")

rep = []
rep += profile(ds).metrics
rep += analyze_missingness(ds).metrics
rep += measure_imbalance(ds, y="label").metrics

final = RunReport(metrics=rep)
final = apply_interpretations(final)
render(final, out_dir="dq_reports")
```

---

## Development

```bash
git clone https://github.com/your-org/dqkit.git
cd dqkit
pip install -e ".[dev]"
pytest
```

Artifacts (histograms, missingness patterns, suspect rows, reports) are written to `artifacts/` or `dq_reports/` directories.

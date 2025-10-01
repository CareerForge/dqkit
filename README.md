
# dqkit

**Data Quality Evaluation Toolkit** (pandas-first)

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


# dqkit Documentation

dqkit is a **pandas-first Data Quality Evaluation Toolkit** providing standardized measures for:

- Noise
- Imbalance
- Redundancy
- Representativeness
- ...and more

## Installation

```bash
pip install dqkit
```

## Getting started

```python
import pandas as pd
from dqkit.io.pandas_io import from_dataframe
from dqkit.profiling import profile

df = pd.DataFrame({"x":[1,2,3]})
ds = from_dataframe(df)
report = profile(ds)
```

See [Usage](usage.md) for details.

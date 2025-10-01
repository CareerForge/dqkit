
# Custom Metrics

You can extend dqkit with your own metrics.

```python
from dqkit.registry import dq_metric
from dqkit.types import Dataset, MetricResult

@dq_metric(id="dq.custom.my_mean")
def my_mean(ds: Dataset) -> MetricResult:
    val = float(ds.df["age"].mean())
    return MetricResult("dq.custom.my_mean","dataset","age",val)
```

```python
from dqkit.registry import compute
result = compute("dq.custom.my_mean", ds)
```

Your metric will appear in reports alongside built-in ones.

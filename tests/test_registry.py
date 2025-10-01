
import pandas as pd
from dqkit.types import Dataset, MetricResult
from dqkit.registry import dq_metric, list_metrics, compute, clear_registry

def setup_function(fn):
    clear_registry()

def test_register_and_compute_simple_metric():
    @dq_metric(id="dq.custom.mean.age")
    def mean_age(ds: Dataset) -> MetricResult:
        v = float(ds.df["age"].mean())
        return MetricResult("dq.custom.mean.age","dataset","age", v, unit="years")

    df = pd.DataFrame({"age":[10,20,30]})
    ds = Dataset(df, name="people")

    # metric appears in registry
    ms = list_metrics()
    assert "dq.custom.mean.age" in ms

    # compute works
    out = compute("dq.custom.mean.age", ds)
    assert out.id == "dq.custom.mean.age"
    assert out.value == 20.0

def test_registry_prefix_filter():
    @dq_metric(id="dq.custom.a")
    def a(ds: Dataset) -> MetricResult:
        return MetricResult("dq.custom.a","dataset","*",1.0)

    @dq_metric(id="dq.other.b")
    def b(ds: Dataset) -> MetricResult:
        return MetricResult("dq.other.b","dataset","*",2.0)

    assert list_metrics(prefix="dq.custom.") == ["dq.custom.a"]

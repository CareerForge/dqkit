
import pandas as pd
from dqkit.types import Dataset, RunReport, MetricResult
from dqkit.slices import evaluate_by_segment

def dummy_profile(ds: Dataset) -> RunReport:
    return RunReport(metrics=[MetricResult('dq.profile.n_rows','dataset','*',len(ds.df))])

def test_evaluate_by_boolean_segment():
    df = pd.DataFrame({'x':[1,2,3,4]})
    ds = Dataset(df, name="toy")
    segs = {'even': df['x'] % 2 == 0}
    rep = evaluate_by_segment(ds, segs, [dummy_profile])
    ids = [m.id for m in rep.metrics]
    assert any(i.endswith('.segment[even]') for i in ids)
    vals = [m.value for m in rep.metrics if m.id.startswith('dq.profile.n_rows')]
    assert set(vals) == {2}  # only 2 even rows

def test_evaluate_by_categorical_segment():
    df = pd.DataFrame({'x':[1,2,3,4], 'g':['A','A','B','B']})
    ds = Dataset(df, name="toy")
    segs = {'group': df['g']}
    rep = evaluate_by_segment(ds, segs, [dummy_profile])
    ids = [m.id for m in rep.metrics]
    assert any('.segment[group=A]' in i for i in ids)
    assert any('.segment[group=B]' in i for i in ids)

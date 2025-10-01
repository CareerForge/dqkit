
import numpy as np
import pandas as pd
from dqkit.types import Dataset
from dqkit.drift import measure_drift, measure_drift_history

def test_measure_drift_basic():
    ref = Dataset(pd.DataFrame({"x": np.arange(100)}), name="ref")
    cur = Dataset(pd.DataFrame({"x": np.arange(100)+5}), name="cur")
    rep = measure_drift(cur, ref, metrics=("psi","ks"))
    m = {mm.id: mm.value for mm in rep.metrics}
    assert "dq.drift.psi.x" in m and "dq.drift.ks.x" in m
    assert "dq.drift.psi.aggregate" in m

def test_measure_drift_history():
    h1 = Dataset(pd.DataFrame({"x": np.arange(50)}), name="t1")
    h2 = Dataset(pd.DataFrame({"x": np.arange(50)+2}), name="t2")
    h3 = Dataset(pd.DataFrame({"x": np.arange(50)+4}), name="t3")
    rep = measure_drift_history([h1, h2, h3], metrics=("psi",))
    m = {mm.id: mm.value for mm in rep.metrics}
    # should have step aggregates and per-feature for last step
    assert "dq.drift.psi.aggregate.step_1" in m and "dq.drift.psi.aggregate.step_2" in m
    assert "dq.drift.psi.x" in m


import numpy as np
import pandas as pd
from dqkit.types import Dataset
from dqkit.representativeness import compare

def test_represent_psi_identical_numeric(tmp_path):
    a = pd.DataFrame({"x": np.arange(100)})
    b = pd.DataFrame({"x": np.arange(100)})
    rep = compare(Dataset(a, name="A"), Dataset(b, name="B"), metrics=("psi","ks"), artifacts_dir=str(tmp_path))
    m = {mm.id: mm.value for mm in rep.metrics}
    assert abs(m["dq.represent.psi.x"]) < 1e-9
    assert m["dq.represent.ks.x"] == 0.0

def test_represent_psi_shifted_numeric():
    a = pd.DataFrame({"x": np.arange(100)})
    b = pd.DataFrame({"x": np.arange(100) + 10})
    rep = compare(Dataset(a, name="A"), Dataset(b, name="B"), metrics=("psi","ks"))
    m = {mm.id: mm.value for mm in rep.metrics}
    assert m["dq.represent.psi.x"] > 0.0
    assert m["dq.represent.ks.x"] > 0.0

def test_represent_categorical():
    a = pd.DataFrame({"c": ["a","a","b","b","b"]})
    b = pd.DataFrame({"c": ["a","a","a","b","b"]})
    rep = compare(Dataset(a, name="A"), Dataset(b, name="B"), features=["c"], metrics=("psi",))
    m = {mm.id: mm.value for mm in rep.metrics}
    assert "dq.represent.psi.c" in m

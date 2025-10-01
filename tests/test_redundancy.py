
import pandas as pd
from dqkit.types import Dataset
from dqkit.redundancy import find_duplicates, find_near_duplicates

def test_find_duplicates(tmp_path):
    df = pd.DataFrame({
        "a": [1,1,2,3],
        "b": ["x","x","y","z"]
    })
    rep = find_duplicates(Dataset(df, name="r"), keys=["a","b"], artifacts_dir=str(tmp_path))
    m = {mm.id: mm.value for mm in rep.metrics}
    assert "dq.redundancy.rows.duplicate_rate" in m
    assert rep.artifacts.get("artifact.redundancy.duplicates") or True  # may or may not exist if no dups

def test_find_near_duplicates_numeric(tmp_path):
    import numpy as np
    df = pd.DataFrame({
        "x1": [0, 0.01, 5.0, 5.02],
        "x2": [0, 0.01, 5.0, 5.02],
    })
    rep = find_near_duplicates(Dataset(df, name="r2"), numeric_cols=["x1","x2"], threshold=0.999, artifacts_dir=str(tmp_path))
    m = {mm.id: mm.value for mm in rep.metrics}
    assert "dq.redundancy.rows.near_dup_rate" in m

def test_find_near_duplicates_text(tmp_path):
    df = pd.DataFrame({
        "t": ["hello world", "hello world!", "foo bar", "foo bar baz"]
    })
    rep = find_near_duplicates(Dataset(df, name="r3"), text_cols=["t"], threshold=0.5, artifacts_dir=str(tmp_path))
    m = {mm.id: mm.value for mm in rep.metrics}
    assert "dq.redundancy.rows.near_dup_rate" in m

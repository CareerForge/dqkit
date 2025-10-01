
import pandas as pd
from dqkit.types import Dataset
from dqkit.missingness import analyze_missingness

def test_missingness_basic(tmp_path):
    df = pd.DataFrame({
        "a": [1, None, 3, None],
        "b": [None, "x", None, "y"],
        "c": [1, 2, 3, 4],
    })
    rep = analyze_missingness(Dataset(df, name="m"), artifacts_dir=str(tmp_path))
    m = {mm.id: mm.value for mm in rep.metrics}
    # per-column
    assert m["dq.missing.rate.a"] == 0.5
    assert m["dq.missing.rate.b"] == 0.5
    assert m["dq.missing.rate.c"] == 0.0
    # row rate: rows 0,1,2,3 each have at least one NA -> 1.0
    assert m["dq.missing.row_rate"] == 1.0
    # co-occurrence between a and b: rows 0 and 2 -> 0.5
    assert any(mid.startswith("dq.missing.cooccur.a.b") for mid in m.keys())
    # artifacts
    assert "artifact.missing.cooccurrence" in rep.artifacts
    assert "artifact.missing.top_patterns" in rep.artifacts

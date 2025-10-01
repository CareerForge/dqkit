
import pandas as pd
from dqkit.types import Dataset
from dqkit.profiling import profile

def test_profile_numeric_and_categorical(tmp_path):
    df = pd.DataFrame({
        "num": [1,2,3,4,5],
        "cat": ["a","a","b","b","c"],
        "dt": pd.to_datetime(["2020-01-01","2020-01-02","2020-01-03", None, None])
    })
    ds = Dataset(df, name="toy")
    rep = profile(ds, artifacts_dir=str(tmp_path))
    ids = {m.id for m in rep.metrics}
    assert "dq.profile.count.num" in ids
    assert "dq.profile.mean.num" in ids
    assert "dq.profile.topk.cat" in ids
    assert "dq.profile.min.dt" in ids and "dq.profile.max.dt" in ids
    # artifacts created
    assert any(k.startswith("artifact.") for k in rep.artifacts)

def test_profile_quantiles_hist(tmp_path):
    import numpy as np
    df = pd.DataFrame({"x": np.arange(100)})
    rep = profile(Dataset(df, name="q"), bins=5, artifacts_dir=str(tmp_path))
    m = {mm.id: mm.value for mm in rep.metrics}
    assert m["dq.profile.q50.x"] == 49.5 or abs(m["dq.profile.q50.x"]-49.5) < 1e-9
    assert "artifact.hist.x" in rep.artifacts

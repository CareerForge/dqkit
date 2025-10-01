
import numpy as np
import pandas as pd
from dqkit.types import Dataset
from dqkit.anomaly import score_outliers

def test_anomaly_auto(tmp_path):
    df = pd.DataFrame({
        "x": [1,2,3,4,5, 100],   # obvious outlier
        "y": [1,1,1,2,2, 2]
    })
    rep = score_outliers(Dataset(df, name="anom"), method="auto", contamination=0.1, artifacts_dir=str(tmp_path))
    m = {mm.id: mm.value for mm in rep.metrics}
    assert m["dq.anomaly.rate"] > 0.0
    assert "artifact.anomaly.rows" in rep.artifacts

def test_anomaly_iforest(tmp_path):
    df = pd.DataFrame({
        "x": np.r_[np.random.normal(0,1,100), [8,9,10]],
        "y": np.r_[np.random.normal(0,1,100), [8,9,10]]
    })
    rep = score_outliers(Dataset(df, name="iforest"), method="iforest", contamination=0.03, artifacts_dir=str(tmp_path))
    m = {mm.id: mm.value for mm in rep.metrics}
    assert m["dq.anomaly.rate"] > 0.0

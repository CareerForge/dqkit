
import numpy as np
import pandas as pd
from dqkit.types import Dataset
from dqkit.noise import estimate_label_noise

def test_noise_with_proba(tmp_path):
    df = pd.DataFrame({
        "x1": [0, 0, 1, 1],
        "x2": [0, 1, 0, 1],
        "y":  [0, 0, 1, 0],  # last one is mislabeled (should be 1)
    })
    # classes [0,1]
    proba = np.array([
        [0.95, 0.05],
        [0.90, 0.10],
        [0.05, 0.95],
        [0.10, 0.90],  # model says class 1 is likely, but y=0 -> high suspicion
    ])
    rep = estimate_label_noise(Dataset(df, name="toy"), y="y", proba=proba, classes=[0,1], artifacts_dir=str(tmp_path))
    m = {mm.id: mm.value for mm in rep.metrics}
    assert "dq.noise.rate.overall" in m
    assert m["dq.noise.suspect_count"] >= 1
    assert "artifact.noise.suspects" in rep.artifacts

def test_noise_1nn_heuristic(tmp_path):
    # Two clusters; one point mislabeled
    df = pd.DataFrame({
        "x": [0.0, 0.1, 5.0, 5.1, 5.2],
        "y": [0,    0,   1,   1,   0],  # last is mislabeled
    })
    rep = estimate_label_noise(Dataset(df, name="nn"), y="y", artifacts_dir=str(tmp_path))
    m = {mm.id: mm.value for mm in rep.metrics}
    assert m["dq.noise.rate.overall"] > 0.0
    assert "dq.noise.rate.class.0" in m and "dq.noise.rate.class.1" in m

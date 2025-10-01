
import pandas as pd
from dqkit.types import Dataset
from dqkit.imbalance import measure_imbalance, simulate_rebalance

def test_measure_imbalance_basic():
    df = pd.DataFrame({"y": [0,0,0,1,1,2]})
    rep = measure_imbalance(Dataset(df, name="imb"), y="y")
    m = {mm.id: mm.value for mm in rep.metrics}
    assert m["dq.imbalance.ir"] >= 1.0
    assert "dq.imbalance.counts" in m
    assert "dq.imbalance.gini" in m
    # rarity available per class value
    keys = [mm.id for mm in rep.metrics if mm.id.startswith("dq.imbalance.rarity.")]
    assert any("dq.imbalance.rarity.0" == k for k in keys)

def test_simulate_rebalance_uniform():
    counts = {"0": 100, "1": 20, "2": 5}
    plan = simulate_rebalance(counts, target="uniform")
    assert plan["total_added"] == (100-20) + (100-5)
    assert plan["target"]["1"] == 100 and plan["target"]["2"] == 100

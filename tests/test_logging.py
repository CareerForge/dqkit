
import os, json, time
import pandas as pd
from dqkit.types import Dataset, MetricResult, RunReport
from dqkit.logging import log_run, load_history, diff_metrics

def test_log_and_load(tmp_path):
    ds = Dataset(pd.DataFrame({'x':[1,2,3]}), name='t')
    rep1 = RunReport(metrics=[MetricResult('dq.profile.n_rows','dataset','*',3)])
    rep2 = RunReport(metrics=[MetricResult('dq.profile.n_rows','dataset','*',4)])
    p1 = log_run(rep1, store=str(tmp_path))
    time.sleep(0.01)
    p2 = log_run(rep2, store=str(tmp_path))
    history = load_history(str(tmp_path))
    assert len(history) == 2
    diff = diff_metrics(history[0], history[1])
    assert diff['dq.profile.n_rows'] == 1.0

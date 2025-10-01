
from dqkit.types import MetricResult, RunReport
from dqkit.checks import Check, run_checks

def test_run_checks():
    rep = RunReport(metrics=[
        MetricResult('dq.drift.psi.aggregate','dataset','*',0.12),
        MetricResult('dq.imbalance.ir','dataset','y',4.0),
    ])
    ch = [
        Check('dq.drift.psi.aggregate','<',0.25),
        Check('dq.imbalance.ir','<=',10.0)
    ]
    results = run_checks(rep, ch)
    assert all(r.passed for r in results)

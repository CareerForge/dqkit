
from dqkit.types import MetricResult, RunReport
from dqkit.standards import interpret, apply_interpretations

def test_interpret_psi_and_vif_and_ir():
    psi_good = MetricResult('dq.represent.psi.age','column','age',0.05)
    psi_bad  = MetricResult('dq.represent.psi.age','column','age',0.3)
    vif_ok   = MetricResult('dq.redundancy.features.vif.x','column','x',5.0)
    vif_bad  = MetricResult('dq.redundancy.features.vif.x','column','x',11.0)
    ir_bad   = MetricResult('dq.imbalance.ir','dataset','y',12.0)

    assert interpret(psi_good).interpretation == 'good'
    assert interpret(psi_bad).interpretation == 'bad'
    assert interpret(vif_ok).interpretation == 'good'
    assert interpret(vif_bad).interpretation == 'bad'
    assert interpret(ir_bad).interpretation == 'bad'

def test_apply_interpretations():
    rep = RunReport(metrics=[
        MetricResult('dq.missing.rate.foo','column','foo',0.3),
        MetricResult('dq.redundancy.features.maxcorr.bar','column','bar',0.96),
    ])
    out = apply_interpretations(rep)
    d = {m.id: m.interpretation for m in out.metrics}
    assert d['dq.missing.rate.foo'] == 'bad'
    assert d['dq.redundancy.features.maxcorr.bar'] == 'bad'

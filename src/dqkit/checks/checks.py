
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union
import operator
from ..types import RunReport, MetricResult

_OPS = {
    "<": operator.lt, "<=": operator.le, "==": operator.eq, "!=": operator.ne, ">=": operator.ge, ">": operator.gt
}

@dataclass
class Check:
    metric_id: str
    op: str
    threshold: Union[int, float]
    severity: str = "error"  # or "warn"
    description: Optional[str] = None

@dataclass
class CheckResult:
    check: Check
    passed: bool
    actual: Optional[Union[int, float]]

def run_checks(report: RunReport, checks: List[Check]) -> List[CheckResult]:
    # map metric id -> numeric value (ignore non-numeric for threshold checks)
    mv = {m.id: m.value for m in report.metrics if isinstance(m.value, (int, float))}
    results: List[CheckResult] = []
    for c in checks:
        op_func = _OPS.get(c.op)
        actual = mv.get(c.metric_id, None)
        passed = False if actual is None else bool(op_func(actual, c.threshold))
        results.append(CheckResult(check=c, passed=passed, actual=actual))
    return results

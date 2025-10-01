
# CI Integration

dqkit fits into CI/CD via:

- **pytest checks** — Write `Check` rules and run them in CI. Fail build if thresholds are violated.
- **GitHub Actions** — See `.github/workflows/ci.yml`
- **Artifacts** — HTML/Markdown reports uploaded as build artifacts.

Example check:

```python
from dqkit.checks import Check, run_checks

checks = [Check("dq.drift.psi.aggregate","<",0.25)]
results = run_checks(report, checks)
assert all(r.passed for r in results)
```

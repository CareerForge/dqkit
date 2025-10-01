
import pandas as pd
from dqkit.types import Dataset, MetricResult, RunReport
from dqkit.report import render

def test_render_creates_files(tmp_path):
    rep = RunReport(metrics=[
        MetricResult('dq.profile.n_rows','dataset','*', 123),
        MetricResult('dq.missing.row_rate','dataset','*', 0.1),
        MetricResult('dq.imbalance.ir','dataset','y', 2.0),
    ], artifacts={'artifact.example':'/path/to/file.csv'})
    out = render(rep, out_dir=str(tmp_path), filename="test_report.html")
    assert (tmp_path / "test_report.html").exists()
    assert (tmp_path / "test_report.md").exists()
    # sanity check contents
    html_text = (tmp_path / "test_report.html").read_text()
    assert "Data Quality Report" in html_text
    assert "dq.imbalance.ir" in html_text

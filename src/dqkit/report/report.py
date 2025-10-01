
from __future__ import annotations
from typing import Dict, List, Optional
import os, json, html
from datetime import datetime
from ..types import RunReport, MetricResult

SECTION_ORDER = [
    ("Schema & Validation", "dq.validation."),
    ("Profiling", "dq.profile."),
    ("Missingness", "dq.missing."),
    ("Label Noise", "dq.noise."),
    ("Imbalance", "dq.imbalance."),
    ("Redundancy (Rows)", "dq.redundancy.rows."),
    ("Redundancy (Features)", "dq.redundancy.features."),
    ("Representativeness", "dq.represent."),
    ("Drift", "dq.drift."),
    ("Anomaly", "dq.anomaly."),
]

def _group_metrics(metrics: List[MetricResult]) -> Dict[str, List[MetricResult]]:
    groups: Dict[str, List[MetricResult]] = {title: [] for title, _ in SECTION_ORDER}
    groups.setdefault("Other", [])
    for m in metrics:
        placed = False
        for title, prefix in SECTION_ORDER:
            if m.id.startswith(prefix):
                groups[title].append(m); placed = True; break
        if not placed:
            groups["Other"].append(m)
    return groups

def _to_html_table(metrics: List[MetricResult]) -> str:
    rows = []
    rows.append("<tr><th>Metric ID</th><th>Level</th><th>Target</th><th>Value</th><th>Unit</th></tr>")
    for m in metrics:
        val = html.escape(json.dumps(m.value, default=str)) if not isinstance(m.value, (int,float,str)) else html.escape(str(m.value))
        rows.append(f"<tr><td><code>{html.escape(m.id)}</code></td><td>{html.escape(m.level)}</td><td>{html.escape(str(m.target))}</td><td>{val}</td><td>{html.escape(m.unit or '')}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"

def render(report: RunReport, out_dir: str = "dq_reports", filename: Optional[str] = None, title: str = "Data Quality Report") -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    fname = filename or "report.html"
    html_path = os.path.join(out_dir, fname)
    md_path = os.path.join(out_dir, (fname[:-5] if fname.endswith('.html') else fname) + ".md")

    grouped = _group_metrics(report.metrics)

    # HTML
    sections_html = []
    for title_sec, _ in SECTION_ORDER + [("Other", "")]:
        ms = grouped.get(title_sec, [])
        if not ms: continue
        sections_html.append(f"<h2>{html.escape(title_sec)}</h2>" + _to_html_table(ms))
    artifacts_html = ""
    if report.artifacts:
        arts = "".join(f"<li><code>{html.escape(k)}</code>: {html.escape(v)}</li>" for k,v in report.artifacts.items())
        artifacts_html = f"<h2>Artifacts</h2><ul>{arts}</ul>"

    html_doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
    h1 {{ margin-bottom: 0; }}
    .meta {{ color: #666; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    th {{ background: #f7f7f7; }}
    code {{ background: #f1f1f1; padding: 1px 4px; border-radius: 3px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="meta">Generated: {html.escape(ts)}</div>
  {''.join(sections_html)}
  {artifacts_html}
</body>
</html>"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    # Markdown (simple)
    md_lines = [f"# {title}", f"_Generated: {ts}_", ""]
    for title_sec, _ in SECTION_ORDER + [("Other", "")]:
        ms = grouped.get(title_sec, [])
        if not ms: continue
        md_lines.append(f"## {title_sec}")
        md_lines.append("| Metric ID | Level | Target | Value | Unit |")
        md_lines.append("|---|---|---|---:|---|")
        for m in ms:
            val = json.dumps(m.value, default=str) if not isinstance(m.value, (int,float,str)) else str(m.value)
            md_lines.append(f"| `{m.id}` | {m.level} | {m.target} | {val} | {m.unit or ''} |")
        md_lines.append("")
    if report.artifacts:
        md_lines.append("## Artifacts")
        for k, v in report.artifacts.items():
            md_lines.append(f"- `{k}`: {v}")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return {"html": html_path, "markdown": md_path}

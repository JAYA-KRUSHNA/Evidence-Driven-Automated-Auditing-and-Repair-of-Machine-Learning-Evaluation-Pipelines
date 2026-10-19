"""
Report Builder — Generates structured audit reports.

Produces JSON and HTML reports with:
- Dataset provenance and fingerprint
- Experiment configuration
- All findings with evidence
- Before/after comparison
- Software versions and random seeds
"""

import logging
import json
import time
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

from src.config import ARTIFACTS_DIR, get_software_versions

logger = logging.getLogger(__name__)


class ReportBuilder:
    """Builds structured audit reports from experiment and audit results."""

    def build_report(
        self,
        dataset_info: dict,
        experiment_config: dict,
        experiment_result: dict,
        audit_result: dict,
        correction_result: dict = None,
        benchmark_result: dict = None,
    ) -> dict:
        """
        Build a complete audit report.

        Returns a dict that can be exported as JSON or rendered as HTML.
        """
        report = {
            "report_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "report_version": "1.0",
                "software_versions": get_software_versions(),
            },
            "dataset": {
                "name": dataset_info.get("name", "Unknown"),
                "source": dataset_info.get("source", "Unknown"),
                "source_url": dataset_info.get("source_url", ""),
                "row_count": dataset_info.get("row_count", 0),
                "col_count": dataset_info.get("col_count", 0),
                "checksum": dataset_info.get("checksum", ""),
                "license": dataset_info.get("license_info", ""),
            },
            "experiment": {
                "name": experiment_config.get("experiment_name", ""),
                "task_type": experiment_config.get("task_type", ""),
                "target_column": experiment_config.get("target_column", ""),
                "models": experiment_config.get("model_names", []),
                "split_strategy": experiment_config.get("split_strategy", ""),
                "test_size": experiment_config.get("test_size", 0.2),
                "cv_folds": experiment_config.get("cv_folds", 5),
                "random_seed": experiment_config.get("random_seed", 42),
            },
            "results": experiment_result,
            "audit": audit_result,
        }

        if correction_result:
            report["corrections"] = correction_result

        if benchmark_result:
            report["benchmark"] = benchmark_result

        report["limitations"] = [
            "Detection is limited to supported leakage patterns.",
            "Code analysis covers scikit-learn patterns; other frameworks may not be detected.",
            "Near-duplicate detection uses sampling and may miss some cases.",
            "Temporal and group column detection is heuristic-based.",
            "Auto-corrections are limited to supported, safe operations.",
        ]

        return report

    def export_json(self, report: dict, filename: str = "") -> Path:
        """Export report as JSON."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audit_report_{timestamp}.json"

        filepath = ARTIFACTS_DIR / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"JSON report saved to {filepath}")
        return filepath

    def export_html(self, report: dict, filename: str = "") -> Path:
        """Export report as HTML."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audit_report_{timestamp}.html"

        filepath = ARTIFACTS_DIR / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        html_content = self._render_html(report)

        with open(filepath, "w") as f:
            f.write(html_content)

        logger.info(f"HTML report saved to {filepath}")
        return filepath

    def _render_html(self, report: dict) -> str:
        """Render report as HTML using inline template."""
        dataset = report.get("dataset", {})
        experiment = report.get("experiment", {})
        audit = report.get("audit", {})
        metadata = report.get("report_metadata", {})
        corrections = report.get("corrections", {})
        limitations = report.get("limitations", [])

        findings_html = ""
        findings = audit.get("findings", [])
        if isinstance(findings, list):
            for f in findings:
                if isinstance(f, dict):
                    severity = f.get("severity", "info")
                    sev_color = {
                        "critical": "#dc2626", "high": "#ea580c",
                        "medium": "#ca8a04", "low": "#2563eb", "info": "#6b7280"
                    }.get(severity, "#6b7280")

                    findings_html += f"""
                    <div class="finding" style="border-left: 4px solid {sev_color};">
                        <div class="finding-header">
                            <span class="severity" style="background:{sev_color};">{severity.upper()}</span>
                            <span class="finding-id">{f.get('finding_id', '')}</span>
                            <span class="status">{f.get('status', '')}</span>
                        </div>
                        <p class="description">{f.get('description', '')}</p>
                        <p class="explanation"><em>{f.get('explanation', '')}</em></p>
                        <p class="recommendation"><strong>Recommendation:</strong> {f.get('recommendation', '')}</p>
                        <p class="confidence">Confidence: {f.get('confidence', 0):.0%}</p>
                    </div>
                    """

        comparison_html = ""
        if corrections:
            before = corrections.get("before_metrics", {})
            after = corrections.get("after_metrics", {})
            if before or after:
                comparison_html = "<h2>Before/After Comparison</h2><table><tr><th>Metric</th><th>Before</th><th>After</th><th>Change</th></tr>"
                all_keys = set(list(before.keys()) + list(after.keys()))
                for k in sorted(all_keys):
                    if k == "confusion_matrix":
                        continue
                    b = before.get(k, "—")
                    a = after.get(k, "—")
                    if isinstance(b, (int, float)) and isinstance(a, (int, float)):
                        change = a - b
                        comparison_html += f"<tr><td>{k}</td><td>{b:.4f}</td><td>{a:.4f}</td><td>{change:+.4f}</td></tr>"
                    else:
                        comparison_html += f"<tr><td>{k}</td><td>{b}</td><td>{a}</td><td>—</td></tr>"
                comparison_html += "</table>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Experiment Audit Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #f8fafc; color: #1e293b; line-height: 1.6; padding: 2rem; max-width: 1000px; margin: 0 auto; }}
        h1 {{ color: #0f172a; margin-bottom: 0.5rem; }}
        h2 {{ color: #334155; margin: 2rem 0 1rem; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; }}
        .meta {{ color: #64748b; font-size: 0.9rem; margin-bottom: 2rem; }}
        table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
        th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #f1f5f9; font-weight: 600; }}
        .finding {{ background: white; border-radius: 8px; padding: 1rem; margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .finding-header {{ display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; }}
        .severity {{ color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }}
        .finding-id {{ font-family: monospace; color: #64748b; }}
        .status {{ color: #64748b; font-style: italic; }}
        .description {{ margin: 0.5rem 0; }}
        .explanation {{ color: #475569; font-size: 0.9rem; }}
        .recommendation {{ margin-top: 0.5rem; font-size: 0.9rem; }}
        .confidence {{ color: #94a3b8; font-size: 0.85rem; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0; }}
        .summary-card {{ background: white; border-radius: 8px; padding: 1rem; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .summary-card .value {{ font-size: 2rem; font-weight: 700; color: #0f172a; }}
        .summary-card .label {{ color: #64748b; font-size: 0.85rem; }}
    </style>
</head>
<body>
    <h1>🔍 ML Experiment Audit Report</h1>
    <p class="meta">Generated: {metadata.get('generated_at', '')} | Report v{metadata.get('report_version', '1.0')}</p>

    <h2>Dataset</h2>
    <table>
        <tr><th>Property</th><th>Value</th></tr>
        <tr><td>Name</td><td>{dataset.get('name', '')}</td></tr>
        <tr><td>Source</td><td>{dataset.get('source', '')}</td></tr>
        <tr><td>Rows</td><td>{dataset.get('row_count', '')}</td></tr>
        <tr><td>Columns</td><td>{dataset.get('col_count', '')}</td></tr>
        <tr><td>License</td><td>{dataset.get('license', '')}</td></tr>
    </table>

    <h2>Experiment Configuration</h2>
    <table>
        <tr><th>Property</th><th>Value</th></tr>
        <tr><td>Task Type</td><td>{experiment.get('task_type', '')}</td></tr>
        <tr><td>Target Column</td><td>{experiment.get('target_column', '')}</td></tr>
        <tr><td>Models</td><td>{', '.join(experiment.get('models', []))}</td></tr>
        <tr><td>Split Strategy</td><td>{experiment.get('split_strategy', '')}</td></tr>
        <tr><td>Test Size</td><td>{experiment.get('test_size', '')}</td></tr>
        <tr><td>Random Seed</td><td>{experiment.get('random_seed', '')}</td></tr>
    </table>

    <h2>Audit Summary</h2>
    <div class="summary-grid">
        <div class="summary-card">
            <div class="value">{audit.get('total_findings', 0)}</div>
            <div class="label">Total Findings</div>
        </div>
        <div class="summary-card">
            <div class="value" style="color:#dc2626;">{audit.get('confirmed', 0)}</div>
            <div class="label">Confirmed</div>
        </div>
        <div class="summary-card">
            <div class="value" style="color:#ca8a04;">{audit.get('suspected', 0)}</div>
            <div class="label">Suspected</div>
        </div>
        <div class="summary-card">
            <div class="value" style="color:#2563eb;">{audit.get('auto_correctable', 0)}</div>
            <div class="label">Auto-Correctable</div>
        </div>
    </div>

    <h2>Findings</h2>
    {findings_html if findings_html else '<p>No findings.</p>'}

    {comparison_html}

    <h2>Limitations</h2>
    <ul>
        {''.join(f'<li>{l}</li>' for l in limitations)}
    </ul>

    <h2>Software Versions</h2>
    <table>
        {''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in metadata.get('software_versions', {}).items())}
    </table>
</body>
</html>"""

        return html

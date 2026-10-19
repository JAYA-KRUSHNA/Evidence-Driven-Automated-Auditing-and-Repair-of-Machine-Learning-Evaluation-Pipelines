"""Reports Page — Generate and download audit reports."""

import streamlit as st
import json
from pathlib import Path

from src.reports.report_builder import ReportBuilder
from src.config import ARTIFACTS_DIR


def render_reports():
    st.markdown("# 📄 Audit Reports")
    st.markdown("*Generate and download reproducible audit reports with full evidence.*")

    experiment_result = st.session_state.get("experiment_result")
    audit_result = st.session_state.get("audit_result")
    experiment_config = st.session_state.get("experiment_config")
    correction_result = st.session_state.get("correction_result")
    benchmark_result = st.session_state.get("benchmark_result")

    if not experiment_result or not audit_result:
        st.info("Run an experiment and audit first from the '🧪 New Experiment' page.")
        return

    builder = ReportBuilder()

    # Build dataset info
    dataset_info = {
        "name": st.session_state.get("exp_ds_name", "Unknown"),
        "source": getattr(st.session_state.get("exp_ds_meta"), "source", "unknown"),
        "source_url": getattr(st.session_state.get("exp_ds_meta"), "source_url", ""),
        "row_count": experiment_result.profile.get("row_count", 0) if experiment_result.profile else 0,
        "col_count": experiment_result.profile.get("col_count", 0) if experiment_result.profile else 0,
        "license_info": getattr(st.session_state.get("exp_ds_meta"), "license_info", ""),
    }

    # Build experiment result summary
    exp_results = {}
    for model_name, tr in experiment_result.training_results.items():
        exp_results[model_name] = {
            "test_metrics": {k: v for k, v in tr.test_metrics.items()
                           if isinstance(v, (int, float)) and k != "confusion_matrix"},
            "train_metrics": {k: v for k, v in tr.train_metrics.items()
                            if isinstance(v, (int, float)) and k != "confusion_matrix"},
            "cv_metrics": {k: round(v, 4) for k, v in tr.cv_metrics.items()},
            "training_time": tr.training_time,
        }

    # Build audit summary
    audit_summary = audit_result.summary()
    audit_summary["findings"] = [f.to_dict() for f in audit_result.findings]

    # Build correction summary
    correction_summary = None
    if correction_result and correction_result.status == "completed":
        correction_summary = correction_result.comparison_summary()

    # Build benchmark summary
    benchmark_summary = None
    if benchmark_result:
        benchmark_summary = benchmark_result.to_dict()

    # Generate report
    report = builder.build_report(
        dataset_info=dataset_info,
        experiment_config=experiment_config.to_dict() if experiment_config else {},
        experiment_result=exp_results,
        audit_result=audit_summary,
        correction_result=correction_summary,
        benchmark_result=benchmark_summary,
    )

    # Preview
    st.markdown('<div class="section-header">📋 Report Preview</div>', unsafe_allow_html=True)

    # Summary section
    st.markdown(f"""
    <div class="glass-panel">
        <h3 style="color: #f1f5f9;">Report Summary</h3>
        <p>📊 <strong>Dataset:</strong> {dataset_info['name']} ({dataset_info.get('row_count', 0):,} rows)</p>
        <p>🧪 <strong>Task:</strong> {experiment_config.task_type if experiment_config else ''} | Target: {experiment_config.target_column if experiment_config else ''}</p>
        <p>🔍 <strong>Findings:</strong> {audit_summary['total_findings']} total ({audit_summary['confirmed']} confirmed, {audit_summary['suspected']} suspected)</p>
        <p>🔧 <strong>Correctable:</strong> {audit_summary['auto_correctable']} issues</p>
        <p>⏱️ <strong>Audit Time:</strong> {audit_summary.get('duration_seconds', 0):.1f}s</p>
    </div>
    """, unsafe_allow_html=True)

    # Export buttons
    st.markdown('<div class="section-header">💾 Export</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Generate JSON Report", use_container_width=True):
            filepath = builder.export_json(report)
            st.success(f"✅ JSON report saved to: {filepath}")

            # Provide download
            with open(filepath, "r") as f:
                json_str = f.read()
            st.download_button(
                "⬇️ Download JSON",
                data=json_str,
                file_name=filepath.name,
                mime="application/json",
                use_container_width=True,
            )

    with col2:
        if st.button("📥 Generate HTML Report", use_container_width=True):
            filepath = builder.export_html(report)
            st.success(f"✅ HTML report saved to: {filepath}")

            with open(filepath, "r") as f:
                html_str = f.read()
            st.download_button(
                "⬇️ Download HTML",
                data=html_str,
                file_name=filepath.name,
                mime="text/html",
                use_container_width=True,
            )

    # Raw JSON view
    with st.expander("🔍 Raw Report JSON"):
        st.json(report)

    # Existing reports
    st.markdown('<div class="section-header">📂 Saved Reports</div>', unsafe_allow_html=True)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    existing_reports = sorted(ARTIFACTS_DIR.glob("audit_report_*"), reverse=True)

    if existing_reports:
        for report_path in existing_reports[:10]:
            col1, col2 = st.columns([3, 1])
            with col1:
                size_kb = report_path.stat().st_size / 1024
                st.markdown(f"📄 **{report_path.name}** ({size_kb:.1f} KB)")
            with col2:
                with open(report_path, "r") as f:
                    content = f.read()
                st.download_button(
                    "⬇️",
                    data=content,
                    file_name=report_path.name,
                    key=f"dl_{report_path.name}",
                )
    else:
        st.info("No saved reports yet. Generate one above!")

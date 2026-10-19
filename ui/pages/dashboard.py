"""Dashboard Page — Home with real statistics from the database."""

import streamlit as st

from src.database.models import get_session, init_db
from src.database.repository import (
    DatasetRepository, ExperimentRepository,
    AuditRunRepository, FindingRepository,
)


def render_dashboard():
    st.markdown("# 🏠 Dashboard")
    st.markdown("*Evidence-driven ML experiment auditing — all data is real.*")

    init_db()
    session = get_session()

    # Real statistics from database
    dataset_count = DatasetRepository.count(session)
    experiment_count = ExperimentRepository.count(session)
    completed_count = ExperimentRepository.count(session, status="completed")
    audit_count = AuditRunRepository.count(session)
    confirmed_findings = FindingRepository.count_confirmed(session)
    suspected_findings = FindingRepository.count_suspected(session)

    session.close()

    # Metric cards
    cols = st.columns(6)

    metrics = [
        ("📚", dataset_count, "Datasets"),
        ("🧪", experiment_count, "Experiments"),
        ("✅", completed_count, "Completed"),
        ("🔍", audit_count, "Audits"),
        ("🔴", confirmed_findings, "Confirmed Issues"),
        ("🟡", suspected_findings, "Suspected Issues"),
    ]

    for col, (icon, value, label) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 1.5rem;">{icon}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick actions
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="glass-panel">
            <h3>🚀 Quick Start</h3>
            <p>Upload a dataset or select from the public library to begin an experiment.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📚 Browse Datasets", use_container_width=True):
            st.session_state["nav_to"] = "datasets"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="glass-panel">
            <h3>🧪 New Experiment</h3>
            <p>Configure and run an ML experiment with automatic auditing.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🧪 Start Experiment", use_container_width=True):
            st.session_state["nav_to"] = "experiment"
            st.rerun()

    with col3:
        st.markdown("""
        <div class="glass-panel">
            <h3>🎯 Benchmark</h3>
            <p>Evaluate the auditor's detection accuracy with controlled tests.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🎯 Run Benchmark", use_container_width=True):
            st.session_state["nav_to"] = "benchmark"
            st.rerun()

    # Recent experiments
    st.markdown('<div class="section-header">📋 Recent Experiments</div>', unsafe_allow_html=True)

    session = get_session()
    recent = ExperimentRepository.get_recent(session, limit=10)
    session.close()

    if recent:
        for exp in recent:
            status_icon = {"completed": "✅", "failed": "❌", "running": "⏳", "pending": "⏸️"}.get(exp.status, "❓")
            st.markdown(f"""
            <div class="glass-panel" style="padding: 0.75rem 1.25rem; margin: 0.5rem 0;">
                <span style="font-size: 1.2rem;">{status_icon}</span>
                <strong style="color: #e2e8f0;">{exp.name}</strong>
                <span style="color: #64748b; margin-left: 1rem;">
                    {exp.task_type} • Target: {exp.target_column}
                    {f' • {exp.duration_seconds:.1f}s' if exp.duration_seconds else ''}
                </span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No experiments yet. Start by uploading a dataset or browsing the library!")

"""Audit Findings Page — View all findings with filters and evidence."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


def render_audit_findings():
    st.markdown("# 🔍 Audit Findings")
    st.markdown("*Detailed view of all detected issues with evidence and recommendations.*")

    audit_result = st.session_state.get("audit_result")

    if audit_result is None:
        st.info("No audit results yet. Run an experiment from the '🧪 New Experiment' page first.")
        return

    findings = audit_result.findings
    summary = audit_result.summary()

    # Summary cards
    cols = st.columns(5)
    metric_data = [
        ("Total", summary["total_findings"], "#818cf8"),
        ("Critical", summary["critical"], "#dc2626"),
        ("High", summary["high"], "#ea580c"),
        ("Confirmed", summary["confirmed"], "#f87171"),
        ("Correctable", summary["auto_correctable"], "#34d399"),
    ]

    for col, (label, value, color) in zip(cols, metric_data):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="background: {color}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Category distribution chart
    by_category = audit_result.findings_by_category()
    if by_category:
        cat_counts = {k.replace("_", " ").title(): len(v) for k, v in by_category.items()}

        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                names=list(cat_counts.keys()),
                values=list(cat_counts.values()),
                title="Findings by Category",
                color_discrete_sequence=px.colors.sequential.Purp,
                hole=0.4,
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            by_severity = audit_result.findings_by_severity()
            sev_counts = {k.title(): len(v) for k, v in by_severity.items()}
            sev_colors = {"Critical": "#dc2626", "High": "#ea580c", "Medium": "#ca8a04", "Low": "#2563eb", "Info": "#6b7280"}

            fig = px.bar(
                x=list(sev_counts.keys()),
                y=list(sev_counts.values()),
                title="Findings by Severity",
                color=list(sev_counts.keys()),
                color_discrete_map=sev_colors,
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Filters
    st.markdown('<div class="section-header">📋 All Findings</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        sev_filter = st.multiselect(
            "Severity",
            ["critical", "high", "medium", "low", "info"],
            default=["critical", "high", "medium", "low", "info"],
        )
    with col2:
        cat_filter = st.multiselect(
            "Category",
            list(set(f.category for f in findings)),
            default=list(set(f.category for f in findings)),
        )
    with col3:
        status_filter = st.multiselect(
            "Status",
            list(set(f.status for f in findings)),
            default=list(set(f.status for f in findings)),
        )

    filtered = [
        f for f in findings
        if f.severity in sev_filter
        and f.category in cat_filter
        and f.status in status_filter
    ]

    st.markdown(f"**Showing {len(filtered)} of {len(findings)} findings**")

    # Display findings
    for finding in filtered:
        sev_class = f"severity-{finding.severity}"
        status_class = f"status-{finding.status}"

        with st.container():
            st.markdown(f"""
            <div class="glass-panel">
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem;">
                    <span class="{sev_class}">{finding.severity.upper()}</span>
                    <code style="color: #818cf8;">{finding.finding_id}</code>
                    <span class="{status_class}">{finding.status}</span>
                    <span style="color: #64748b; font-size: 0.8rem;">
                        {finding.category.replace('_', ' ').title()}
                    </span>
                    {'<span style="color: #34d399; font-size: 0.8rem;">🔧 Auto-correctable</span>' if finding.auto_correctable else ''}
                </div>
                <p style="color: #f1f5f9; font-weight: 600;">{finding.description}</p>
                <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem;">
                    <em>{finding.explanation}</em>
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Expandable details
            with st.expander(f"📋 Details — {finding.finding_id}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Recommendation:**")
                    st.markdown(f"> {finding.recommendation}")
                    st.markdown(f"**Confidence:** {finding.confidence:.0%}")

                    if finding.affected_columns:
                        st.markdown(f"**Affected Columns:** `{'`, `'.join(finding.affected_columns)}`")

                with col2:
                    if finding.evidence:
                        st.markdown("**Evidence:**")
                        evidence = finding.evidence if isinstance(finding.evidence, dict) else {}
                        st.json(evidence)

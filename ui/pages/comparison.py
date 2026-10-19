"""Comparison Page — Before/After correction comparison."""

import streamlit as st
import plotly.graph_objects as go
import json

from src.repair.correction_engine import CorrectionEngine, CorrectionPlan


def render_comparison():
    st.markdown("# 📊 Before / After Comparison")
    st.markdown("*Compare original and corrected pipelines. A lower corrected score may indicate a more honest evaluation.*")

    experiment_result = st.session_state.get("experiment_result")
    audit_result = st.session_state.get("audit_result")
    experiment_config = st.session_state.get("experiment_config")
    df = st.session_state.get("experiment_df")

    if not all([experiment_result, audit_result, experiment_config, df is not None]):
        st.info("Run an experiment first from the '🧪 New Experiment' page.")
        return

    # Show correctable findings
    correctable = [f for f in audit_result.findings if f.auto_correctable]

    if not correctable:
        st.success("✅ No auto-correctable issues found. The experiment appears clean!")
        return

    st.markdown(f"**{len(correctable)} auto-correctable findings detected:**")

    for f in correctable:
        sev_class = f"severity-{f.severity}"
        st.markdown(f"""
        <div class="glass-panel" style="padding: 0.75rem 1rem; margin: 0.25rem 0;">
            <span class="{sev_class}">{f.severity.upper()}</span>
            <strong style="color: #e2e8f0; margin-left: 0.5rem;">[{f.finding_id}]</strong>
            <span style="color: #cbd5e1; margin-left: 0.5rem;">{f.description}</span>
            <br>
            <span style="color: #34d399; font-size: 0.85rem;">🔧 Fix: {f.recommendation}</span>
        </div>
        """, unsafe_allow_html=True)

    # Correction
    st.markdown('<div class="section-header">🔧 Apply Corrections</div>', unsafe_allow_html=True)

    st.warning(
        "⚠️ Corrections will create a **separate corrected experiment**. "
        "Original data and results are preserved."
    )

    if st.button("🔧 Apply Corrections & Rerun", type="primary", use_container_width=True):
        engine = CorrectionEngine()

        # Propose corrections
        plan = engine.propose_corrections(
            audit_result, df,
            experiment_config.target_column,
            split_config={"strategy": experiment_config.split_strategy},
        )

        if not plan.has_corrections:
            st.info("No corrections could be generated.")
            return

        st.markdown("**Corrections to apply:**")
        for action in plan.actions:
            st.markdown(f"- 🔧 **{action.correction_type}**: {action.description}")

        # Apply
        progress = st.progress(0, text="Applying corrections...")

        def progress_cb(step, total, msg):
            progress.progress(step / total, text=msg)

        correction_result = engine.apply_corrections(
            plan=plan,
            df=df,
            config=experiment_config,
            original_result=experiment_result,
            original_audit=audit_result,
            progress_callback=progress_cb,
        )

        st.session_state["correction_result"] = correction_result
        progress.progress(1.0, text="✅ Complete!")

        if correction_result.status == "failed":
            st.error(f"Correction failed: {correction_result.error}")
            return

        st.success("✅ Corrections applied and experiment rerun!")

    # Display comparison if available
    correction_result = st.session_state.get("correction_result")

    if correction_result and correction_result.status == "completed":
        st.markdown('<div class="section-header">📊 Results Comparison</div>', unsafe_allow_html=True)

        before = correction_result.before_metrics
        after = correction_result.after_metrics

        # Filter to comparable metrics
        compare_keys = [
            k for k in set(list(before.keys()) + list(after.keys()))
            if k != "confusion_matrix" and isinstance(before.get(k, 0), (int, float))
            and isinstance(after.get(k, 0), (int, float))
        ]

        if compare_keys:
            # Comparison chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Original",
                x=[k.replace("_", " ").title() for k in compare_keys],
                y=[before.get(k, 0) for k in compare_keys],
                marker_color="rgba(239, 68, 68, 0.7)",
            ))
            fig.add_trace(go.Bar(
                name="Corrected",
                x=[k.replace("_", " ").title() for k in compare_keys],
                y=[after.get(k, 0) for k in compare_keys],
                marker_color="rgba(52, 211, 153, 0.7)",
            ))
            fig.update_layout(
                barmode="group",
                title="Original vs Corrected Metrics",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Table
            st.markdown("**Metric Details:**")
            rows = []
            for k in sorted(compare_keys):
                b = before.get(k, 0)
                a = after.get(k, 0)
                change = a - b
                direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                rows.append({
                    "Metric": k.replace("_", " ").title(),
                    "Original": f"{b:.4f}",
                    "Corrected": f"{a:.4f}",
                    "Change": f"{direction} {change:+.4f}",
                })

            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Resolution summary
        st.markdown('<div class="section-header">📋 Resolution Summary</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="glass-panel" style="text-align:center;">
                <div class="metric-value" style="background: #34d399; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {len(correction_result.resolved_findings)}
                </div>
                <div class="metric-label">Resolved</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;">
                <div class="metric-value" style="background: #f87171; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {len(correction_result.unresolved_findings)}
                </div>
                <div class="metric-label">Unresolved</div>
            </div>
            """, unsafe_allow_html=True)

        st.info(
            "💡 A decrease in model performance after correction typically means "
            "the original evaluation was artificially inflated by data leakage. "
            "The corrected score is more honest and trustworthy."
        )

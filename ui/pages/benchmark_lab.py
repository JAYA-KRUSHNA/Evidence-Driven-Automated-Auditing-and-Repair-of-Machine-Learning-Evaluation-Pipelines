"""Benchmark Lab Page — Test the auditor's own detection accuracy."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data.dataset_catalog import DatasetCatalog
from src.data.dataset_loader import DatasetLoader, DatasetLoadError
from src.benchmark.benchmark_runner import BenchmarkRunner


def render_benchmark_lab():
    st.markdown("# 🎯 Benchmark Lab")
    st.markdown("*Evaluate the auditor's detection accuracy using controlled corruption experiments on real data.*")

    loader = DatasetLoader()

    # Dataset selection for benchmark
    st.markdown('<div class="section-header">Select Base Dataset</div>', unsafe_allow_html=True)

    catalog = DatasetCatalog()
    dataset_options = {f"{d.name} ({d.source})": d for d in catalog.get_all()}
    selected = st.selectbox("Base dataset for corruption", list(dataset_options.keys()))
    ds_meta = dataset_options[selected]

    col1, col2 = st.columns(2)
    with col1:
        task_type = st.selectbox("Task type", ["classification", "regression"])
    with col2:
        target_col = st.text_input("Target column", value=ds_meta.target_column or "")

    if st.button("🎯 Run Full Benchmark", type="primary", use_container_width=True):
        if not target_col:
            st.error("Specify a target column.")
            return

        with st.spinner(f"Loading {ds_meta.name}..."):
            try:
                df, _ = loader.load(ds_meta.source, ds_meta.source_id, ds_meta.name)
            except DatasetLoadError as e:
                st.error(f"Failed to load dataset: {e}")
                return

        if target_col not in df.columns:
            st.error(f"Column '{target_col}' not found. Available: {', '.join(df.columns[:20])}")
            return

        st.info(f"Running benchmark with {len(df):,} rows × {len(df.columns)} columns...")

        runner = BenchmarkRunner()
        progress = st.progress(0, text="Starting benchmark...")

        def progress_cb(step, total, msg):
            progress.progress(step / total, text=msg)

        result = runner.run_benchmark(
            df, target_col, task_type,
            model_name="random_forest",
            progress_callback=progress_cb,
        )

        st.session_state["benchmark_result"] = result
        progress.progress(1.0, text="✅ Complete!")

    # Display results
    result = st.session_state.get("benchmark_result")
    if result is None:
        st.info("Select a dataset and run the benchmark to evaluate the auditor.")
        return

    st.markdown('<div class="section-header">📊 Aggregate Results</div>', unsafe_allow_html=True)

    # Overall metrics
    cols = st.columns(4)
    overall = [
        ("Precision", result.overall_precision, "#34d399"),
        ("Recall", result.overall_recall, "#818cf8"),
        ("F1 Score", result.overall_f1, "#c084fc"),
        ("FP Rate", result.overall_false_positive_rate, "#f87171"),
    ]

    for col, (label, value, color) in zip(cols, overall):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="background: {color}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {value:.1%}
                </div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"**Total time:** {result.total_time:.1f}s | **Scenarios:** {len(result.scenario_results)}")

    # Per-scenario results table
    st.markdown('<div class="section-header">📋 Per-Scenario Results</div>', unsafe_allow_html=True)

    rows = []
    for r in result.scenario_results:
        status = "✅" if r.error is None else "❌"
        rows.append({
            "Status": status,
            "Scenario": r.scenario_name,
            "Type": r.corruption_type,
            "Ground Truth": ", ".join(r.ground_truth) if r.ground_truth else "Clean",
            "Detected": ", ".join(r.detected_categories) if r.detected_categories else "None",
            "TP": len(r.true_positives),
            "FP": len(r.false_positives),
            "FN": len(r.false_negatives),
            "Precision": f"{r.precision:.2f}",
            "Recall": f"{r.recall:.2f}",
            "F1": f"{r.f1:.2f}",
            "Holdout": "🔒" if r.is_holdout else "",
        })

    df_results = pd.DataFrame(rows)
    st.dataframe(df_results, use_container_width=True, hide_index=True)

    # Visualization
    col1, col2 = st.columns(2)

    with col1:
        # F1 by scenario
        scenario_names = [r.scenario_name for r in result.scenario_results if r.error is None]
        f1_scores = [r.f1 for r in result.scenario_results if r.error is None]

        fig = px.bar(
            x=scenario_names, y=f1_scores,
            title="F1 Score by Scenario",
            color=f1_scores,
            color_continuous_scale="RdYlGn",
            labels={"x": "Scenario", "y": "F1 Score"},
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd5e1",
            xaxis_tickangle=-45,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Per-category performance
        if result.per_category:
            cats = list(result.per_category.keys())
            precisions = [result.per_category[c].get("precision", 0) for c in cats]
            recalls = [result.per_category[c].get("recall", 0) for c in cats]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Precision",
                x=[c.replace("_", " ").title() for c in cats],
                y=precisions,
                marker_color="rgba(52, 211, 153, 0.7)",
            ))
            fig.add_trace(go.Bar(
                name="Recall",
                x=[c.replace("_", " ").title() for c in cats],
                y=recalls,
                marker_color="rgba(129, 140, 248, 0.7)",
            ))
            fig.update_layout(
                barmode="group",
                title="Detection by Category",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Holdout analysis
    holdout = [r for r in result.scenario_results if r.is_holdout and r.error is None]
    if holdout:
        st.markdown('<div class="section-header">🔒 Holdout Scenario Results</div>', unsafe_allow_html=True)
        st.info("Holdout scenarios test the auditor on patterns it was NOT specifically designed to detect.")
        for h in holdout:
            detected = "✅ Detected" if h.recall > 0 else "❌ Missed"
            st.markdown(f"**{h.scenario_name}**: {detected} (F1={h.f1:.2f})")

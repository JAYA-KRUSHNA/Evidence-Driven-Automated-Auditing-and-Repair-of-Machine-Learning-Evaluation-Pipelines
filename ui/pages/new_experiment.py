"""New Experiment Page — Configure and run ML experiments with real training."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data.dataset_catalog import DatasetCatalog, CURATED_CATALOG
from src.data.dataset_loader import DatasetLoader, DatasetLoadError
from src.data.dataset_profiler import DatasetProfiler
from src.experiments.experiment_runner import ExperimentRunner, ExperimentConfig
from src.experiments.pipeline_builder import PipelineBuilder
from src.experiments.metrics import MetricsCalculator
from src.audit.audit_engine import AuditEngine
from src.database.models import get_session, init_db
from src.database.repository import DatasetRepository


def render_new_experiment():
    st.markdown("# 🧪 New Experiment")
    st.markdown("*Train real ML models and automatically audit for evaluation errors.*")

    loader = DatasetLoader()
    profiler = DatasetProfiler()

    # ── Step 1: Select Dataset ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Step 1: Select Dataset</div>', unsafe_allow_html=True)

    data_source = st.radio(
        "Data source",
        ["📚 Public Dataset", "📁 Upload CSV"],
        horizontal=True,
    )

    df = None
    dataset_name = ""
    dataset_id = ""

    if data_source == "📚 Public Dataset":
        catalog = DatasetCatalog()
        dataset_options = {f"{d.name} ({d.source})": d for d in catalog.get_all()}

        selected = st.selectbox("Choose dataset", list(dataset_options.keys()))
        ds_meta = dataset_options[selected]

        if st.button("📥 Load Dataset", type="primary"):
            with st.spinner(f"Loading {ds_meta.name}..."):
                try:
                    df, metadata = loader.load(ds_meta.source, ds_meta.source_id, ds_meta.name)
                    st.session_state["exp_df"] = df
                    st.session_state["exp_ds_name"] = ds_meta.name
                    st.session_state["exp_ds_meta"] = ds_meta

                    # Register in DB
                    init_db()
                    session = get_session()
                    existing = DatasetRepository.get_by_source_id(session, ds_meta.source, ds_meta.source_id)
                    if not existing:
                        db_ds = DatasetRepository.create(
                            session,
                            name=ds_meta.name, source=ds_meta.source,
                            source_id=ds_meta.source_id, source_url=ds_meta.source_url,
                            task_type=ds_meta.task_type, target_column=ds_meta.target_column,
                            row_count=len(df), col_count=len(df.columns),
                            missing_pct=round(df.isna().mean().mean() * 100, 2),
                            license_info=ds_meta.license_info, description=ds_meta.description,
                        )
                        st.session_state["exp_ds_id"] = db_ds.id
                    else:
                        st.session_state["exp_ds_id"] = existing.id
                    session.close()

                    st.success(f"✅ Loaded: {len(df):,} rows × {len(df.columns)} cols")
                except DatasetLoadError as e:
                    st.error(f"Failed to load: {e}")
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv", "parquet"], key="exp_upload")
        if uploaded:
            try:
                df, metadata = loader.load_from_upload(uploaded)
                st.session_state["exp_df"] = df
                st.session_state["exp_ds_name"] = uploaded.name
                st.session_state["exp_ds_id"] = metadata.get("source_id", "uploaded")
                st.success(f"✅ Loaded: {len(df):,} rows × {len(df.columns)} cols")
            except DatasetLoadError as e:
                st.error(f"Upload failed: {e}")

    # Load from session state
    if "exp_df" in st.session_state:
        df = st.session_state["exp_df"]
        dataset_name = st.session_state.get("exp_ds_name", "")
        dataset_id = st.session_state.get("exp_ds_id", "")

    if df is None:
        st.info("Select or upload a dataset to continue.")
        return

    # Show preview
    with st.expander("📋 Dataset Preview", expanded=False):
        st.dataframe(df.head(50), use_container_width=True)

    # Profile
    profile = profiler.profile(df)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{profile.row_count:,}")
    col2.metric("Columns", profile.col_count)
    col3.metric("Missing", f"{profile.missing_cells_pct}%")
    col4.metric("Duplicates", profile.duplicate_rows)

    if profile.warnings:
        for w in profile.warnings:
            st.warning(w)

    # ── Step 2: Configure Experiment ───────────────────────────────────────────
    st.markdown('<div class="section-header">Step 2: Configure Experiment</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        target_column = st.selectbox(
            "🎯 Target Column",
            profile.target_candidates if profile.target_candidates else list(df.columns),
            help="The column you want to predict."
        )

        # Analyze target
        target_analysis = profiler.get_target_analysis(df, target_column)
        detected_task = target_analysis.get("task_type", "classification")

        task_type = st.selectbox(
            "📋 Task Type",
            ["classification", "regression"],
            index=0 if detected_task == "classification" else 1,
            help=f"Auto-detected: {detected_task}"
        )

    with col2:
        available_models = PipelineBuilder.get_available_models(task_type)
        selected_models = st.multiselect(
            "🤖 Models to Train",
            list(available_models.keys()),
            default=["random_forest"],
            format_func=lambda x: available_models[x],
        )

        random_seed = st.number_input("🌱 Random Seed", value=42, min_value=0, max_value=99999)

    col3, col4 = st.columns(2)
    with col3:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    with col4:
        cv_folds = st.slider("CV Folds", 2, 10, 5)

    # Target analysis display
    if target_analysis:
        with st.expander("🎯 Target Analysis"):
            if "class_distribution" in target_analysis:
                dist = target_analysis["class_distribution"]
                fig = px.bar(
                    x=list(dist.keys()), y=list(dist.values()),
                    labels={"x": "Class", "y": "Count"},
                    title="Class Distribution",
                    color_discrete_sequence=["#818cf8"],
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#cbd5e1",
                )
                st.plotly_chart(fig, use_container_width=True)

                if target_analysis.get("is_imbalanced"):
                    st.warning("⚠️ Target is imbalanced. Consider stratified splitting and F1/PR-AUC metrics.")

    # ── Step 3: Run Experiment ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Step 3: Run Experiment</div>', unsafe_allow_html=True)

    experiment_name = st.text_input("Experiment Name", value=f"{dataset_name}_experiment")

    if st.button("🚀 Run Experiment & Audit", type="primary", use_container_width=True):
        if not selected_models:
            st.error("Select at least one model.")
            return

        config = ExperimentConfig(
            dataset_id=dataset_id,
            target_column=target_column,
            task_type=task_type,
            model_names=selected_models,
            test_size=test_size,
            cv_folds=cv_folds,
            random_seed=random_seed,
            experiment_name=experiment_name,
        )

        # Run experiment
        runner = ExperimentRunner()
        progress_bar = st.progress(0, text="Starting experiment...")

        def progress_cb(step, total, msg):
            progress_bar.progress(step / total, text=msg)

        with st.spinner("Training models and running audit..."):
            result = runner.run(df, config, progress_callback=progress_cb)

        if result.status == "failed":
            st.error(f"❌ Experiment failed: {result.error}")
            return

        st.session_state["experiment_result"] = result
        st.session_state["experiment_config"] = config
        st.session_state["experiment_df"] = df

        progress_bar.progress(1.0, text="✅ Complete!")
        st.success(f"✅ Experiment completed in {result.total_time:.1f}s")

        # Show results
        st.markdown('<div class="section-header">📊 Results</div>', unsafe_allow_html=True)

        for model_name, train_result in result.training_results.items():
            st.markdown(f"### Model: {model_name}")

            if "error" in train_result.test_metrics:
                st.error(f"Training failed: {train_result.test_metrics['error']}")
                continue

            # Metrics table
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Test Metrics**")
                metrics_df = MetricsCalculator.format_metrics_table(
                    train_result.test_metrics, task_type
                )
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            with col2:
                st.markdown("**Baseline Comparison**")
                if train_result.baseline_metrics:
                    baseline_df = MetricsCalculator.format_metrics_table(
                        train_result.baseline_metrics, task_type
                    )
                    st.dataframe(baseline_df, use_container_width=True, hide_index=True)

            # Confusion matrix
            if "confusion_matrix" in train_result.test_metrics:
                cm = train_result.test_metrics["confusion_matrix"]
                fig = px.imshow(
                    cm, text_auto=True,
                    title=f"Confusion Matrix — {model_name}",
                    color_continuous_scale="Blues",
                    labels={"x": "Predicted", "y": "Actual"},
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#cbd5e1",
                    width=400, height=400,
                )
                st.plotly_chart(fig, use_container_width=False)

            # Feature importances
            if train_result.feature_importances:
                top_features = dict(list(train_result.feature_importances.items())[:15])
                fig = px.bar(
                    x=list(top_features.values()),
                    y=list(top_features.keys()),
                    orientation="h",
                    title=f"Top Feature Importances — {model_name}",
                    color_discrete_sequence=["#818cf8"],
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#cbd5e1",
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True)

        # Run audit
        st.markdown('<div class="section-header">🔍 Running Audit...</div>', unsafe_allow_html=True)

        audit_engine = AuditEngine()
        audit_progress = st.progress(0, text="Starting audit...")

        # Prepare split data for audit
        from src.experiments.split_manager import SplitManager
        from src.experiments.pipeline_builder import PipelineBuilder as PB

        X = df.drop(columns=[target_column]).dropna(subset=[], how="all")
        y = df[target_column]
        df_clean = df.dropna(subset=[target_column])
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]

        if task_type == "classification":
            y, _ = PB.encode_target(y, "classification")

        X_train, X_test, y_train, y_test = SplitManager.train_test_split(
            X, y, test_size=test_size, random_seed=random_seed, task_type=task_type
        )

        # Get metrics from first model
        first_model_result = list(result.training_results.values())[0]

        def audit_progress_cb(step, total, msg):
            audit_progress.progress(step / total, text=msg)

        audit_result = audit_engine.run_full_audit(
            df=df,
            target_column=target_column,
            task_type=task_type,
            experiment_id=result.experiment_id,
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            train_metrics=first_model_result.train_metrics,
            test_metrics=first_model_result.test_metrics,
            cv_metrics=first_model_result.cv_metrics,
            baseline_metrics=first_model_result.baseline_metrics,
            feature_importances=first_model_result.feature_importances,
            target_analysis=result.target_analysis,
            pipeline=first_model_result.pipeline,
            progress_callback=audit_progress_cb,
        )

        st.session_state["audit_result"] = audit_result
        audit_progress.progress(1.0, text="✅ Audit complete!")

        # Show audit summary
        summary = audit_result.summary()
        st.markdown(f"""
        <div class="glass-panel">
            <h3 style="color: #f1f5f9;">Audit Summary</h3>
            <p>Found <strong>{summary['total_findings']}</strong> findings:
            <span class="status-confirmed">{summary['confirmed']} confirmed</span>,
            <span class="status-suspected">{summary['suspected']} suspected</span>,
            <span style="color: #60a5fa;">{summary['auto_correctable']} auto-correctable</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Brief finding list
        for finding in audit_result.findings[:10]:
            sev_class = f"severity-{finding.severity}"
            st.markdown(f"""
            <div class="glass-panel" style="padding: 0.75rem 1rem; margin: 0.25rem 0;">
                <span class="{sev_class}">{finding.severity.upper()}</span>
                <strong style="color: #e2e8f0; margin-left: 0.5rem;">[{finding.finding_id}]</strong>
                <span style="color: #cbd5e1; margin-left: 0.5rem;">{finding.description}</span>
            </div>
            """, unsafe_allow_html=True)

        if len(audit_result.findings) > 10:
            st.info(f"Showing 10 of {len(audit_result.findings)} findings. See the Audit Findings page for all.")

        st.success("✅ Experiment and audit complete! Check 'Audit Findings' and 'Before/After' pages for details.")

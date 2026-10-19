"""Dataset Library Page — Browse, search, and download real public datasets."""

import streamlit as st
import pandas as pd

from src.data.dataset_catalog import DatasetCatalog
from src.data.dataset_loader import DatasetLoader, DatasetLoadError
from src.data.dataset_profiler import DatasetProfiler
from src.database.models import get_session, init_db
from src.database.repository import DatasetRepository


def render_dataset_library():
    st.markdown("# 📚 Dataset Library")
    st.markdown("*Browse and download real public datasets. All data is downloaded from official sources.*")

    # Initialize
    if "catalog" not in st.session_state:
        st.session_state["catalog"] = DatasetCatalog()

    catalog = st.session_state["catalog"]
    loader = DatasetLoader()

    # Search and filters
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        search_query = st.text_input("🔍 Search datasets", placeholder="e.g., cancer, income, wine...")
    with col2:
        source_filter = st.selectbox("Source", ["All"] + catalog.sources)
    with col3:
        task_filter = st.selectbox("Task Type", ["All", "classification", "regression", "both"])

    # Apply filters
    source = "" if source_filter == "All" else source_filter
    task = "" if task_filter == "All" else task_filter
    datasets = catalog.search(query=search_query, source=source, task_type=task)

    st.markdown(f"**{len(datasets)}** datasets found")

    # OpenML search
    with st.expander("🌐 Search OpenML for more datasets"):
        oml_query = st.text_input("Search OpenML", placeholder="e.g., diabetes, heart, housing...")
        if st.button("Search OpenML") and oml_query:
            with st.spinner("Searching OpenML..."):
                results = catalog.search_openml(oml_query, max_results=10)
                if results:
                    st.success(f"Found {len(results)} datasets on OpenML!")
                else:
                    st.warning("No datasets found or OpenML is unavailable.")

    # Dataset cards
    for ds in datasets:
        is_cached = loader.is_cached(ds.source, ds.source_id)
        cache_icon = "✅" if is_cached else "⬇️"
        source_badge = {
            "sklearn": "🔵 sklearn",
            "openml": "🟢 OpenML",
            "uci": "🟠 UCI",
            "upload": "📁 Upload",
        }.get(ds.source, ds.source)

        with st.container():
            st.markdown(f"""
            <div class="glass-panel">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="font-size: 1.1rem; color: #e2e8f0;">{ds.name}</strong>
                        <span style="margin-left: 0.5rem;">{source_badge}</span>
                        <span style="margin-left: 0.5rem;">{cache_icon}</span>
                    </div>
                    <div style="color: #64748b; font-size: 0.85rem;">
                        {ds.row_count:,} rows × {ds.col_count} cols
                        {f' • {ds.missing_pct:.1f}% missing' if ds.missing_pct > 0 else ''}
                    </div>
                </div>
                <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem;">{ds.description[:200]}</p>
                <div style="margin-top: 0.5rem;">
                    <span style="color: #818cf8; font-size: 0.8rem;">Task: {ds.task_type}</span>
                    <span style="color: #64748b; font-size: 0.8rem; margin-left: 1rem;">Target: {ds.target_column or 'Auto-detect'}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_a, col_b, col_c = st.columns([1, 1, 2])

            with col_a:
                if st.button(f"{'✅ Loaded' if is_cached else '⬇️ Download'}", key=f"dl_{ds.source}_{ds.source_id}", disabled=is_cached):
                    with st.spinner(f"Downloading {ds.name}..."):
                        try:
                            df, metadata = loader.load(ds.source, ds.source_id, ds.name)
                            # Store in database
                            init_db()
                            session = get_session()
                            existing = DatasetRepository.get_by_source_id(session, ds.source, ds.source_id)
                            if not existing:
                                from src.data.dataset_cache import DatasetCache
                                cache = DatasetCache()
                                cached_path = cache.get_cached_path(ds.source, ds.source_id)

                                DatasetRepository.create(
                                    session,
                                    name=ds.name,
                                    source=ds.source,
                                    source_id=ds.source_id,
                                    source_url=ds.source_url,
                                    task_type=ds.task_type,
                                    target_column=ds.target_column,
                                    row_count=len(df),
                                    col_count=len(df.columns),
                                    missing_pct=round(df.isna().mean().mean() * 100, 2),
                                    file_path=str(cached_path) if cached_path else "",
                                    license_info=ds.license_info,
                                    description=ds.description,
                                )
                            session.close()
                            st.success(f"✅ Downloaded: {len(df):,} rows × {len(df.columns)} cols")
                            st.rerun()
                        except DatasetLoadError as e:
                            st.error(f"Download failed: {e}")

            with col_b:
                if is_cached and st.button("👁️ Preview", key=f"pv_{ds.source}_{ds.source_id}"):
                    try:
                        df, _ = loader.load(ds.source, ds.source_id, ds.name)
                        st.dataframe(df.head(20), use_container_width=True)

                        # Quick profile
                        profiler = DatasetProfiler()
                        profile = profiler.profile(df, ds.target_column)
                        st.markdown(f"**Rows:** {profile.row_count:,} | **Cols:** {profile.col_count} | **Missing:** {profile.missing_cells_pct}% | **Duplicates:** {profile.duplicate_rows}")
                    except Exception as e:
                        st.error(f"Preview failed: {e}")

            with col_c:
                if ds.source_url:
                    st.markdown(f"[🔗 Source]({ds.source_url})", unsafe_allow_html=True)

    # Upload section
    st.markdown('<div class="section-header">📁 Upload Your Own Dataset</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV file", type=["csv", "parquet"])
    if uploaded:
        try:
            df, metadata = loader.load_from_upload(uploaded)
            st.success(f"✅ Loaded: {len(df):,} rows × {len(df.columns)} cols")
            st.dataframe(df.head(20), use_container_width=True)

            # Store in database
            init_db()
            session = get_session()
            source_id = metadata.get("source_id", "")

            DatasetRepository.create(
                session,
                name=uploaded.name.rsplit(".", 1)[0],
                source="upload",
                source_id=source_id,
                task_type="",
                row_count=len(df),
                col_count=len(df.columns),
                missing_pct=round(df.isna().mean().mean() * 100, 2),
                description=f"User uploaded: {uploaded.name}",
            )
            session.close()

            # Store in session for use in experiments
            st.session_state["uploaded_df"] = df
            st.session_state["uploaded_name"] = uploaded.name

        except DatasetLoadError as e:
            st.error(f"Upload failed: {e}")

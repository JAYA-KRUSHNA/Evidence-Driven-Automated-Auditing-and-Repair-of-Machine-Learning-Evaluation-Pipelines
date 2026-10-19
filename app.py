"""
AI ML Experiment Auditor — Main Streamlit Application

Entry point for the dashboard. Configures Streamlit and handles navigation.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Initialize database
from src.database.models import init_db
init_db()

# Page config
st.set_page_config(
    page_title="AI ML Experiment Auditor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }

    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.5);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    /* Finding severity badges */
    .severity-critical { background: linear-gradient(135deg, #dc2626, #b91c1c); color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; }
    .severity-high { background: linear-gradient(135deg, #ea580c, #c2410c); color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; }
    .severity-medium { background: linear-gradient(135deg, #ca8a04, #a16207); color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; }
    .severity-low { background: linear-gradient(135deg, #2563eb, #1d4ed8); color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; }
    .severity-info { background: rgba(100,116,139,0.3); color: #cbd5e1; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; }

    /* Status badges */
    .status-confirmed { color: #f87171; font-weight: 600; }
    .status-suspected { color: #fbbf24; font-weight: 600; }
    .status-informational { color: #60a5fa; font-weight: 600; }

    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 2rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(129, 140, 248, 0.3);
    }

    /* Glass panel */
    .glass-panel {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin: 1rem 0;
    }

    /* Streamlit overrides */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    h1, h2, h3 { color: #f1f5f9 !important; }
    p, li { color: #cbd5e1; }

    .stSelectbox label, .stMultiSelect label, .stRadio label, .stNumberInput label {
        color: #cbd5e1 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        color: #94a3b8;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.2) !important;
        color: #818cf8 !important;
        border-color: rgba(99, 102, 241, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ────────────────────────────────────────────────────────

st.sidebar.markdown("""
<div style="text-align:center; padding: 1rem 0;">
    <div style="font-size: 2.5rem;">🔬</div>
    <div style="font-size: 1.2rem; font-weight: 700; color: #818cf8;">ML Experiment Auditor</div>
    <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.25rem;">Evidence-Driven Audit Framework</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

pages = {
    "🏠 Dashboard": "dashboard",
    "📚 Dataset Library": "datasets",
    "🧪 New Experiment": "experiment",
    "🔍 Audit Findings": "findings",
    "📊 Before/After": "comparison",
    "🎯 Benchmark Lab": "benchmark",
    "📄 Reports": "reports",
}

selected_page = st.sidebar.radio(
    "Navigation",
    list(pages.keys()),
    label_visibility="collapsed",
)

page_key = pages[selected_page]

# ── Page Router ────────────────────────────────────────────────────────────────

if page_key == "dashboard":
    from ui.pages.dashboard import render_dashboard
    render_dashboard()
elif page_key == "datasets":
    from ui.pages.dataset_library import render_dataset_library
    render_dataset_library()
elif page_key == "experiment":
    from ui.pages.new_experiment import render_new_experiment
    render_new_experiment()
elif page_key == "findings":
    from ui.pages.audit_findings import render_audit_findings
    render_audit_findings()
elif page_key == "comparison":
    from ui.pages.comparison import render_comparison
    render_comparison()
elif page_key == "benchmark":
    from ui.pages.benchmark_lab import render_benchmark_lab
    render_benchmark_lab()
elif page_key == "reports":
    from ui.pages.reports import render_reports
    render_reports()

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align:center; color: #475569; font-size: 0.75rem; padding: 0.5rem;">
    Built for research. All data is real.<br>
    v1.0 • Python + scikit-learn + Streamlit
</div>
""", unsafe_allow_html=True)

"""
EDA Module — Plotly Interactive Visualizations Engine.

Generates dark-mode, glassmorphic Plotly figure specifications for:
- Univariate: Histogram + KDE, Box Plot, Violin Plot, Categorical Count Bar, Donut/Pie Chart
- Bivariate: Scatter + Trendline, Grouped Box Plot, Stacked Bar Chart
- Multivariate: Correlation Heatmap, PCA 2D/3D Scatter, VIF Scores Bar
- Outliers: Highlighted Outlier Scatter, Tukey Bounds
- Target: Class Distribution Bar/Pie, Feature-Target Association
All figures are returned as JSON-serializable dicts compatible with client-side Plotly.js.
"""

from typing import Any
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats


def apply_theme(fig: go.Figure, **kwargs):
    """Apply consistent glassmorphic dark theme without keyword collisions."""
    margin = kwargs.pop("margin", {"l": 50, "r": 30, "t": 50, "b": 50})
    base_layout = {
        "paper_bgcolor": "rgba(0, 0, 0, 0)",
        "plot_bgcolor": "rgba(255, 255, 255, 0.02)",
        "font": {"family": "Inter, -apple-system, sans-serif", "color": "#94a3b8", "size": 12},
        "colorway": ["#38bdf8", "#818cf8", "#34d399", "#f472b6", "#fbbf24", "#a78bfa", "#fb7185"],
        "margin": margin,
        "hoverlabel": {
            "bgcolor": "#0f172a",
            "bordercolor": "rgba(255, 255, 255, 0.15)",
            "font": {"color": "#f8fafc", "family": "Inter, sans-serif"},
        },
    }
    base_layout.update(kwargs)
    fig.update_layout(**base_layout)
    fig.update_xaxes(gridcolor="rgba(255, 255, 255, 0.06)", zerolinecolor="rgba(255, 255, 255, 0.1)")
    fig.update_yaxes(gridcolor="rgba(255, 255, 255, 0.06)", zerolinecolor="rgba(255, 255, 255, 0.1)")


def fig_to_dict(fig: go.Figure) -> dict[str, Any]:
    """Convert Plotly figure to JSON-safe dictionary."""
    return fig.to_plotly_json()


# ── Univariate Charts ─────────────────────────────────────────────────────────

def chart_histogram_kde(series: pd.Series, col_name: str) -> dict[str, Any]:
    """Histogram with superimposed KDE density curve."""
    clean = series.dropna()
    fig = go.Figure()

    if len(clean) > 0:
        fig.add_trace(go.Histogram(
            x=clean,
            name="Frequency",
            nbinsx=min(35, max(12, int(np.sqrt(len(clean))))),
            marker=dict(color="rgba(56, 189, 248, 0.65)", line=dict(color="#38bdf8", width=1)),
            histnorm="probability density",
            opacity=0.75,
        ))

    # KDE curve
    if len(clean) >= 10 and clean.std() > 0:
        try:
            kde = stats.gaussian_kde(clean if len(clean) <= 2000 else clean.sample(2000, random_state=42))
            x_pts = np.linspace(clean.min(), clean.max(), 120)
            y_pts = kde.evaluate(x_pts)
            fig.add_trace(go.Scatter(
                x=x_pts,
                y=y_pts,
                mode="lines",
                name="KDE Density",
                line=dict(color="#34d399", width=2.5),
            ))
        except Exception:
            pass

    apply_theme(
        fig,
        title=f"Distribution & Density: {col_name}",
        xaxis_title=col_name,
        yaxis_title="Density",
        showlegend=True,
        bargap=0.05,
    )
    return fig_to_dict(fig)


def chart_boxplot(series: pd.Series, col_name: str) -> dict[str, Any]:
    """Interactive Box plot with jittered outlier points."""
    clean = series.dropna()
    fig = go.Figure()

    fig.add_trace(go.Box(
        y=clean,
        name=col_name,
        boxpoints="outliers",
        jitter=0.3,
        pointpos=-1.8,
        marker=dict(color="#818cf8", size=5, opacity=0.7),
        line=dict(color="#38bdf8", width=2),
        fillcolor="rgba(129, 140, 248, 0.25)",
    ))

    apply_theme(
        fig,
        title=f"Box Plot: {col_name}",
        yaxis_title=col_name,
        showlegend=False,
    )
    return fig_to_dict(fig)


def chart_violin(series: pd.Series, col_name: str) -> dict[str, Any]:
    """Violin plot showing kernel density contour and internal box."""
    clean = series.dropna()
    fig = go.Figure()

    fig.add_trace(go.Violin(
        y=clean,
        name=col_name,
        box_visible=True,
        meanline_visible=True,
        fillcolor="rgba(56, 189, 248, 0.3)",
        line_color="#38bdf8",
        points="outliers",
    ))

    apply_theme(
        fig,
        title=f"Violin Plot: {col_name}",
        yaxis_title=col_name,
        showlegend=False,
    )
    return fig_to_dict(fig)


def chart_category_counts(series: pd.Series, col_name: str, max_cats: int = 15) -> dict[str, Any]:
    """Categorical count bar chart."""
    vc = series.dropna().astype(str).value_counts().head(max_cats)
    labels = [str(x) for x in vc.index]
    counts = [int(x) for x in vc.values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=counts,
        marker=dict(
            color=counts,
            colorscale="Viridis",
            line=dict(color="rgba(255,255,255,0.1)", width=1),
        ),
        text=counts,
        textposition="auto",
    ))

    apply_theme(
        fig,
        title=f"Category Frequencies: {col_name}",
        xaxis_title=col_name,
        yaxis_title="Frequency",
        showlegend=False,
    )
    return fig_to_dict(fig)


def chart_category_donut(series: pd.Series, col_name: str, max_cats: int = 10) -> dict[str, Any]:
    """Donut / Pie chart for categorical distribution."""
    vc = series.dropna().astype(str).value_counts()
    top_vc = vc.head(max_cats)
    labels = [str(x) for x in top_vc.index]
    counts = [int(x) for x in top_vc.values]

    other = int(vc.iloc[max_cats:].sum()) if len(vc) > max_cats else 0
    if other > 0:
        labels.append("Other")
        counts.append(other)

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=counts,
        hole=0.45,
        textinfo="label+percent",
        marker=dict(colors=["#38bdf8", "#818cf8", "#34d399", "#f472b6", "#fbbf24", "#a78bfa", "#fb7185"]),
    )])

    apply_theme(
        fig,
        title=f"Proportion: {col_name}",
        showlegend=True,
    )
    return fig_to_dict(fig)


# ── Bivariate Charts ──────────────────────────────────────────────────────────

def chart_scatter_trendline(df: pd.DataFrame, col1: str, col2: str, sample_size: int = 600) -> dict[str, Any]:
    """Bivariate scatter plot with OLS linear trendline."""
    sub = df[[col1, col2]].dropna()
    sampled = sub if len(sub) <= sample_size else sub.sample(sample_size, random_state=42)

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=sampled[col1],
        y=sampled[col2],
        mode="markers",
        name="Observations",
        marker=dict(
            color="#38bdf8",
            size=6,
            opacity=0.65,
            line=dict(color="rgba(255,255,255,0.2)", width=0.5),
        ),
    ))

    # Trendline
    if len(sub) >= 2:
        try:
            slope, intercept, r_val, _, _ = stats.linregress(sub[col1], sub[col2])
            x_trend = np.linspace(sub[col1].min(), sub[col1].max(), 50)
            y_trend = slope * x_trend + intercept
            fig.add_trace(go.Scatter(
                x=x_trend,
                y=y_trend,
                mode="lines",
                name=f"OLS Trend (R²={r_val**2:.3f})",
                line=dict(color="#f43f5e", width=2.5, dash="dash"),
            ))
        except Exception:
            pass

    apply_theme(
        fig,
        title=f"{col1} vs {col2}",
        xaxis_title=col1,
        yaxis_title=col2,
        showlegend=True,
    )
    return fig_to_dict(fig)


def chart_grouped_box(df: pd.DataFrame, num_col: str, cat_col: str, max_groups: int = 10) -> dict[str, Any]:
    """Grouped box plot: Numerical distribution across categories."""
    sub = df[[num_col, cat_col]].dropna()
    top_cats = sub[cat_col].astype(str).value_counts().head(max_groups).index.tolist()
    sub_filtered = sub[sub[cat_col].astype(str).isin(top_cats)]

    fig = go.Figure()
    for cat in top_cats:
        vals = sub_filtered[sub_filtered[cat_col].astype(str) == cat][num_col]
        fig.add_trace(go.Box(
            y=vals,
            name=str(cat),
            boxpoints="outliers",
        ))

    apply_theme(
        fig,
        title=f"{num_col} by {cat_col}",
        xaxis_title=cat_col,
        yaxis_title=num_col,
        showlegend=False,
    )
    return fig_to_dict(fig)


def chart_stacked_bar(df: pd.DataFrame, col1: str, col2: str, max_cats: int = 8) -> dict[str, Any]:
    """Stacked bar chart for two categorical variables."""
    sub = df[[col1, col2]].dropna().astype(str)
    top1 = sub[col1].value_counts().head(max_cats).index.tolist()
    top2 = sub[col2].value_counts().head(max_cats).index.tolist()
    filtered = sub[sub[col1].isin(top1) & sub[col2].isin(top2)]

    if len(filtered) == 0:
        fig = go.Figure()
        apply_theme(fig, title=f"{col1} vs {col2}")
        return fig_to_dict(fig)

    ct = pd.crosstab(filtered[col1], filtered[col2], normalize="index") * 100.0

    fig = go.Figure()
    for col_val in ct.columns:
        fig.add_trace(go.Bar(
            name=str(col_val),
            x=[str(x) for x in ct.index],
            y=ct[col_val],
        ))

    apply_theme(
        fig,
        barmode="stack",
        title=f"{col1} vs {col2} (Percentage Distribution)",
        xaxis_title=col1,
        yaxis_title="Percentage (%)",
        showlegend=True,
    )
    return fig_to_dict(fig)


# ── Multivariate Charts ───────────────────────────────────────────────────────

def chart_correlation_heatmap(corr_matrix_dict: dict[str, dict[str, float | None]], method: str = "Pearson") -> dict[str, Any]:
    """Interactive correlation matrix heatmap."""
    cols = list(corr_matrix_dict.keys())
    fig = go.Figure()

    if not cols:
        apply_theme(fig, title=f"{method} Feature Correlation Matrix (Insufficient data)")
        return fig_to_dict(fig)

    z_values = [[corr_matrix_dict[r].get(c, 0.0) or 0.0 for c in cols] for r in cols]
    text_values = [[f"{val:.2f}" for val in row] for row in z_values]

    fig.add_trace(go.Heatmap(
        z=z_values,
        x=cols,
        y=cols,
        text=text_values,
        texttemplate="%{text}",
        textfont={"size": 11, "color": "white"},
        colorscale=[
            [0.0, "#f43f5e"],    # -1.0 Strong negative (rose)
            [0.5, "#0f172a"],    #  0.0 Neutral / No correlation (slate-900)
            [1.0, "#10b981"],    # +1.0 Strong positive (emerald)
        ],
        zmin=-1.0,
        zmax=1.0,
        hoverongaps=False,
    ))

    apply_theme(
        fig,
        title=f"{method} Feature Correlation Matrix",
        margin={"l": 80, "r": 50, "t": 60, "b": 80},
    )
    return fig_to_dict(fig)


def chart_pca_projection(pca_data: dict[str, Any], labels: list[str] | None = None) -> dict[str, Any]:
    """2D PCA projection scatter plot."""
    points = pca_data.get("projection_2d", [])
    fig = go.Figure()
    if not points:
        apply_theme(fig, title="PCA 2D Projection (No data)")
        return fig_to_dict(fig)

    x_vals = [p["pc1"] for p in points]
    y_vals = [p["pc2"] for p in points]
    pc1_pct = pca_data.get("pc1_pct", 0.0)
    pc2_pct = pca_data.get("pc2_pct", 0.0)

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="markers",
        marker=dict(
            size=6,
            color="#818cf8",
            opacity=0.75,
            line=dict(color="rgba(255,255,255,0.2)", width=0.5),
        ),
    ))

    apply_theme(
        fig,
        title="PCA 2D Projection",
        xaxis_title=f"PC 1 ({pc1_pct:.1f}% explained)",
        yaxis_title=f"PC 2 ({pc2_pct:.1f}% explained)",
        showlegend=False,
    )
    return fig_to_dict(fig)


def chart_vif_bars(vif_dict: dict[str, float | None]) -> dict[str, Any]:
    """Horizontal bar chart for Variance Inflation Factor (VIF)."""
    fig = go.Figure()
    items = [(k, v) for k, v in vif_dict.items() if v is not None]
    if not items:
        apply_theme(fig, title="Multicollinearity Audit (VIF) - Insufficient features")
        return fig_to_dict(fig)

    items.sort(key=lambda x: x[1])
    features = [k for k, _ in items]
    vifs = [v for _, v in items]
    colors = ["#f43f5e" if v >= 10 else "#fbbf24" if v >= 5 else "#34d399" for v in vifs]

    fig.add_trace(go.Bar(
        x=vifs,
        y=features,
        orientation="h",
        marker=dict(color=colors),
        text=[f"{v:.1f}" for v in vifs],
        textposition="auto",
    ))

    fig.add_vline(x=5.0, line_dash="dot", line_color="#fbbf24", annotation_text="Moderate (5.0)")
    fig.add_vline(x=10.0, line_dash="dash", line_color="#f43f5e", annotation_text="Severe (10.0)")

    apply_theme(
        fig,
        title="Multicollinearity Audit (Variance Inflation Factor - VIF)",
        xaxis_title="VIF Score",
        yaxis_title="Feature",
        margin={"l": 100, "r": 40, "t": 60, "b": 50},
    )
    return fig_to_dict(fig)


# ── Outlier Charts ────────────────────────────────────────────────────────────

def chart_outlier_scatter(series: pd.Series, col_name: str, lower_bound: float, upper_bound: float) -> dict[str, Any]:
    """Scatter of points highlighting inliers vs outliers."""
    clean = series.dropna().reset_index(drop=True)
    inliers = clean[(clean >= lower_bound) & (clean <= upper_bound)]
    outliers = clean[(clean < lower_bound) | (clean > upper_bound)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=inliers.index,
        y=inliers,
        mode="markers",
        name="Inliers",
        marker=dict(color="#38bdf8", size=5, opacity=0.6),
    ))

    if not outliers.empty:
        fig.add_trace(go.Scatter(
            x=outliers.index,
            y=outliers,
            mode="markers",
            name="Outliers",
            marker=dict(color="#f43f5e", size=8, symbol="x"),
        ))

    fig.add_hline(y=upper_bound, line_dash="dash", line_color="#fbbf24", annotation_text=f"Upper Bound: {upper_bound:.2f}")
    fig.add_hline(y=lower_bound, line_dash="dash", line_color="#fbbf24", annotation_text=f"Lower Bound: {lower_bound:.2f}")

    apply_theme(
        fig,
        title=f"Outlier Boundary Plot: {col_name}",
        xaxis_title="Index",
        yaxis_title=col_name,
        showlegend=True,
    )
    return fig_to_dict(fig)

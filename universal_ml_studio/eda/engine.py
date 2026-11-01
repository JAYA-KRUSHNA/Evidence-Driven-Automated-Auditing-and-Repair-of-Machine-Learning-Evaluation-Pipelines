"""
EDA Engine — Automated, Intelligent Exploratory Data Analysis.

Produces comprehensive data profiling, deep quality audits, statistical summaries,
univariate/bivariate/multivariate visualizations, multi-method outlier detection,
target variable analysis, automated feature selection, and data-driven AI insights.

Public API:
----------
    EDAEngine.analyze(df, metadata, target_column=None, task_type=None)
        → dict  (all legacy & enhanced analysis modules combined)
    EDAEngine.save(result, dataset_id, session=None)
        → list[EDAResult]
    EDAEngine.get_overview(df, metadata)
    EDAEngine.get_quality(df, metadata)
    EDAEngine.get_univariate(df, metadata, column=None)
    EDAEngine.get_bivariate(df, metadata, col1=None, col2=None)
    EDAEngine.get_multivariate(df, metadata)
    EDAEngine.get_outliers(df, metadata, method="all")
    EDAEngine.get_target(df, metadata, target_column=None, task_type=None)
    EDAEngine.get_features(df, metadata, target_column=None)
    EDAEngine.get_insights(df, metadata, target_column=None)
    EDAEngine.generate_chart(df, chart_type, params)
    EDAEngine.generate_report(dataset_name, eda_data, format="html")
"""

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from .overview import compute_overview
from .quality import compute_quality
from .univariate import compute_univariate, analyze_numerical_column, analyze_categorical_column
from .bivariate import compute_bivariate
from .multivariate import compute_multivariate
from .outliers import compute_outliers
from .target import compute_target_analysis
from .features import compute_feature_analysis
from .insights import generate_insights
from .charts import (
    chart_histogram_kde,
    chart_boxplot,
    chart_violin,
    chart_category_counts,
    chart_category_donut,
    chart_scatter_trendline,
    chart_grouped_box,
    chart_stacked_bar,
    chart_correlation_heatmap,
    chart_pca_projection,
    chart_vif_bars,
    chart_outlier_scatter,
)
from .reports import generate_html_report, generate_pdf_report, generate_excel_report
from ..config import (
    OUTLIER_IQR_MULTIPLIER,
    OUTLIER_ZSCORE_THRESHOLD,
    CORRELATION_LEAKAGE_THRESHOLD,
)

logger = logging.getLogger(__name__)


class EDAEngine:
    """Run automated, intelligent EDA on any tabular DataFrame."""

    # ── Main Entry Point ──────────────────────────────────────────────────

    @staticmethod
    def analyze(
        df: pd.DataFrame,
        metadata: dict[str, Any] | None = None,
        target_column: str | None = None,
        task_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Run the full EDA pipeline and return a serialisable result dict.
        Guarantees all 7 legacy keys for 100% backward compatibility,
        plus enhanced modules: quality, bivariate, multivariate, outliers, features, insights.
        """
        if metadata is None:
            metadata = {}

        result: dict[str, Any] = {}

        # 1. Overview & Health Score
        result["overview"] = compute_overview(df, metadata)

        # 2. Quality Audit
        result["quality"] = compute_quality(df, metadata)

        # 3. Descriptive Statistics (legacy format preserved)
        result["statistics"] = EDAEngine._statistics(df, metadata)

        # 4. Missing Analysis (legacy format preserved)
        result["missing_analysis"] = EDAEngine._missing_analysis(df)

        # 5. Distribution (legacy format preserved)
        result["distribution"] = EDAEngine._distribution(df, metadata)

        # 6. Correlation (legacy format preserved)
        result["correlation"] = EDAEngine._correlation(df, metadata)

        # 7. Outliers (legacy outlier_summary key preserved)
        outlier_res = compute_outliers(df, metadata, method="all")
        result["outlier_summary"] = {
            "total_iqr_outliers": outlier_res["total_iqr_outliers"],
            "columns_with_outliers": outlier_res["columns_with_outliers"],
            "per_column": outlier_res["per_column"],
        }
        result["outliers"] = outlier_res

        # 8. Bivariate Analysis
        result["bivariate"] = compute_bivariate(df, metadata)

        # 9. Multivariate Analysis (PCA, VIF, Interactions)
        result["multivariate"] = compute_multivariate(df, metadata)

        # 10. Target Analysis
        if target_column and target_column in df.columns:
            result["target_analysis"] = compute_target_analysis(df, target_column, task_type, metadata)
        else:
            suggested = compute_target_analysis(df, None, None, metadata)
            result["target_analysis"] = {
                "available": False,
                "reason": "No target column specified",
                "suggested_targets": suggested.get("suggested_targets", []),
            }

        # 11. Feature Analysis
        result["features"] = compute_feature_analysis(df, target_column, metadata)

        # 12. Intelligent AI Insights Engine
        univariate_full = compute_univariate(df, metadata)
        result["univariate"] = univariate_full
        result["insights"] = generate_insights(
            overview_data=result["overview"],
            quality_data=result["quality"],
            univariate_data=univariate_full,
            correlation_data=result["correlation"],
            outliers_data=result["outlier_summary"],
            target_data=result["target_analysis"],
            features_data=result["features"],
        )

        logger.info("Enhanced EDA complete for dataset with shape %s", df.shape)
        return result

    # ── Modular Sub-APIs ──────────────────────────────────────────────────

    @staticmethod
    def get_overview(df: pd.DataFrame, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return compute_overview(df, metadata)

    @staticmethod
    def get_quality(df: pd.DataFrame, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return compute_quality(df, metadata)

    @staticmethod
    def get_univariate(df: pd.DataFrame, metadata: dict[str, Any] | None = None, column: str | None = None) -> dict[str, Any]:
        return compute_univariate(df, metadata, column=column)

    @staticmethod
    def get_bivariate(df: pd.DataFrame, metadata: dict[str, Any] | None = None, col1: str | None = None, col2: str | None = None) -> dict[str, Any]:
        return compute_bivariate(df, metadata, col1=col1, col2=col2)

    @staticmethod
    def get_multivariate(df: pd.DataFrame, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return compute_multivariate(df, metadata)

    @staticmethod
    def get_outliers(df: pd.DataFrame, metadata: dict[str, Any] | None = None, method: str = "all") -> dict[str, Any]:
        return compute_outliers(df, metadata, method=method)

    @staticmethod
    def get_target(df: pd.DataFrame, metadata: dict[str, Any] | None = None, target_column: str | None = None, task_type: str | None = None) -> dict[str, Any]:
        return compute_target_analysis(df, target_column=target_column, task_type=task_type, metadata=metadata)

    @staticmethod
    def get_features(df: pd.DataFrame, metadata: dict[str, Any] | None = None, target_column: str | None = None) -> dict[str, Any]:
        return compute_feature_analysis(df, target_column=target_column, metadata=metadata)

    @staticmethod
    def get_insights(df: pd.DataFrame, metadata: dict[str, Any] | None = None, target_column: str | None = None) -> list[dict[str, Any]]:
        full = EDAEngine.analyze(df, metadata=metadata, target_column=target_column)
        return full.get("insights", [])

    @staticmethod
    def generate_chart(df: pd.DataFrame, chart_type: str, params: dict[str, Any]) -> dict[str, Any]:
        """Generate interactive Plotly figure based on chart type and parameters."""
        col1 = params.get("col1") or params.get("column")
        col2 = params.get("col2")

        if chart_type == "hist_kde" and col1 and col1 in df.columns:
            return chart_histogram_kde(df[col1], col1)
        elif chart_type == "box" and col1 and col1 in df.columns:
            return chart_boxplot(df[col1], col1)
        elif chart_type == "violin" and col1 and col1 in df.columns:
            return chart_violin(df[col1], col1)
        elif chart_type == "category_bar" and col1 and col1 in df.columns:
            return chart_category_counts(df[col1], col1)
        elif chart_type == "category_donut" and col1 and col1 in df.columns:
            return chart_category_donut(df[col1], col1)
        elif chart_type == "scatter" and col1 and col2 and col1 in df.columns and col2 in df.columns:
            return chart_scatter_trendline(df, col1, col2)
        elif chart_type == "grouped_box" and col1 and col2 and col1 in df.columns and col2 in df.columns:
            return chart_grouped_box(df, num_col=col1, cat_col=col2)
        elif chart_type == "stacked_bar" and col1 and col2 and col1 in df.columns and col2 in df.columns:
            return chart_stacked_bar(df, col1, col2)
        elif chart_type == "corr_heatmap":
            corr_res = EDAEngine._correlation(df, {})
            matrix = corr_res.get("matrix")
            if not matrix:
                num_df = df.select_dtypes(include=[np.number])
                if len(num_df.columns) >= 2:
                    cm = num_df.corr()
                    matrix = {
                        col: {o: _safe_float(cm.loc[col, o]) for o in num_df.columns}
                        for col in num_df.columns
                    }
                else:
                    matrix = {}
            return chart_correlation_heatmap(matrix or {})
        elif chart_type == "pca":
            multi = compute_multivariate(df, {})
            return chart_pca_projection(multi.get("pca", {}))
        elif chart_type == "vif":
            multi = compute_multivariate(df, {})
            return chart_vif_bars(multi.get("vif", {}).get("values", {}))
        elif chart_type == "outlier_scatter" and col1 and col1 in df.columns:
            s = df[col1].dropna()
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return chart_outlier_scatter(s, col1, lower, upper)
        else:
            return {"error": f"Unsupported or misconfigured chart_type '{chart_type}'"}

    @staticmethod
    def generate_report(dataset_name: str, eda_data: dict[str, Any], format: str = "html") -> tuple[bytes | str, str, str]:
        """
        Generate full EDA report in HTML, PDF, or Excel format.
        Returns: (content, media_type, filename)
        """
        fmt = format.lower()
        safe_name = dataset_name.replace(" ", "_").replace("/", "_")

        if fmt == "html":
            content = generate_html_report(dataset_name, eda_data)
            return content, "text/html", f"EDA_Report_{safe_name}.html"
        elif fmt == "pdf":
            content = generate_pdf_report(dataset_name, eda_data)
            return content, "application/pdf", f"EDA_Report_{safe_name}.pdf"
        elif fmt in ("excel", "xlsx"):
            content = generate_excel_report(dataset_name, eda_data)
            return content, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", f"EDA_Report_{safe_name}.xlsx"
        else:
            raise ValueError(f"Unsupported report format '{format}'. Choose from 'html', 'pdf', 'excel'.")

    # ── Save to DB ────────────────────────────────────────────────────────

    @staticmethod
    def save(
        result: dict[str, Any],
        dataset_id: str,
        session=None,
    ):
        """Save each EDA module as a separate EDAResult row."""
        from ..database.models import EDAResult
        from ..database.connection import get_session

        own_session = session is None
        if own_session:
            session = get_session()

        records = []
        try:
            for analysis_type, data in result.items():
                row = EDAResult(
                    dataset_id=dataset_id,
                    analysis_type=analysis_type,
                    result_data=data,
                )
                session.add(row)
                records.append(row)

            if own_session:
                session.commit()
            return records

        except Exception:
            if own_session:
                session.rollback()
            raise
        finally:
            if own_session:
                session.close()

    # ═══════════════════════════════════════════════════════════════════════
    #  Legacy Helper Implementations (100% Backwards Compatible)
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _overview(df: pd.DataFrame, metadata: dict) -> dict:
        return compute_overview(df, metadata)

    @staticmethod
    def _statistics(df: pd.DataFrame, metadata: dict) -> dict:
        numerical_cols = [
            c for c in metadata.get("numerical_columns", [])
            if c in df.columns
        ] or [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
        ]
        categorical_cols = [
            c for c in metadata.get("categorical_columns", [])
            if c in df.columns
        ] or [
            c for c in df.columns
            if c not in numerical_cols
        ]

        numerical_stats: dict[str, Any] = {}
        for col in numerical_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            numerical_stats[col] = {
                "count": int(series.count()),
                "missing": int(df[col].isnull().sum()),
                "mean": _safe_float(series.mean()),
                "std": _safe_float(series.std()),
                "min": _safe_float(series.min()),
                "p25": _safe_float(series.quantile(0.25)),
                "median": _safe_float(series.median()),
                "p75": _safe_float(series.quantile(0.75)),
                "max": _safe_float(series.max()),
                "skewness": _safe_float(series.skew()),
                "kurtosis": _safe_float(series.kurtosis()),
                "nunique": int(series.nunique()),
                "dtype": str(df[col].dtype),
                # Compatibility keys
                "25%": _safe_float(series.quantile(0.25)),
                "50%": _safe_float(series.median()),
                "75%": _safe_float(series.quantile(0.75)),
            }

        categorical_stats: dict[str, Any] = {}
        for col in categorical_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            vc = series.value_counts()
            top_values = {str(k): int(v) for k, v in vc.head(10).items()}
            categorical_stats[col] = {
                "count": int(series.count()),
                "missing": int(df[col].isnull().sum()),
                "nunique": int(series.nunique()),
                "top_value": str(vc.index[0]) if len(vc) > 0 else None,
                "top_freq": int(vc.iloc[0]) if len(vc) > 0 else 0,
                "top_10_values": top_values,
                "dtype": str(df[col].dtype),
            }

        return {
            "numerical": numerical_stats,
            "categorical": categorical_stats,
        }

    @staticmethod
    def _missing_analysis(df: pd.DataFrame) -> dict:
        num_rows = len(df)
        per_column: dict[str, Any] = {}
        for col in df.columns:
            missing_count = int(df[col].isnull().sum())
            if missing_count > 0:
                per_column[col] = {
                    "missing_count": missing_count,
                    "missing_pct": _safe_float(missing_count / max(num_rows, 1) * 100),
                    "dtype": str(df[col].dtype),
                }

        rows_with_missing = int(df.isnull().any(axis=1).sum())

        return {
            "total_missing_cells": int(df.isnull().sum().sum()),
            "columns_with_missing": len(per_column),
            "rows_with_missing": rows_with_missing,
            "rows_with_missing_pct": _safe_float(rows_with_missing / max(num_rows, 1) * 100),
            "per_column": per_column,
        }

    @staticmethod
    def _distribution(df: pd.DataFrame, metadata: dict) -> dict:
        numerical_cols = [
            c for c in metadata.get("numerical_columns", [])
            if c in df.columns
        ] or [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
        ]
        categorical_cols = [
            c for c in metadata.get("categorical_columns", [])
            if c in df.columns
        ] or [
            c for c in df.columns
            if c not in numerical_cols
        ]

        numerical_dist: dict[str, Any] = {}
        for col in numerical_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            try:
                counts, edges = np.histogram(series, bins=10)
                histogram = {
                    "counts": counts.tolist(),
                    "bin_edges": [_safe_float(e) for e in edges],
                }
            except Exception:
                histogram = {}

            numerical_dist[col] = {
                "histogram": histogram,
                "percentiles": {
                    "p5":  _safe_float(series.quantile(0.05)),
                    "p10": _safe_float(series.quantile(0.10)),
                    "p25": _safe_float(series.quantile(0.25)),
                    "p50": _safe_float(series.quantile(0.50)),
                    "p75": _safe_float(series.quantile(0.75)),
                    "p90": _safe_float(series.quantile(0.90)),
                    "p95": _safe_float(series.quantile(0.95)),
                },
                "skewness": _safe_float(series.skew()),
                "is_normal": _is_approximately_normal(series),
            }

        categorical_dist: dict[str, Any] = {}
        for col in categorical_cols:
            series = df[col].dropna()
            vc = series.value_counts(normalize=True)
            categorical_dist[col] = {
                "value_frequencies": {str(k): _safe_float(v) for k, v in vc.head(20).items()},
                "num_categories": int(series.nunique()),
                "top_category_pct": _safe_float(vc.iloc[0] * 100) if len(vc) > 0 else 0,
            }

        return {
            "numerical": numerical_dist,
            "categorical": categorical_dist,
        }

    @staticmethod
    def _correlation(df: pd.DataFrame, metadata: dict) -> dict:
        numerical_cols = [
            c for c in metadata.get("numerical_columns", [])
            if c in df.columns
        ] or [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
        ]

        if len(numerical_cols) < 2:
            return {"available": False, "reason": "Need at least 2 numerical columns"}

        num_df = df[numerical_cols].dropna()
        if len(num_df) < 5:
            return {"available": False, "reason": "Insufficient rows after dropping NaN"}

        try:
            corr_matrix = num_df.corr()
        except Exception as exc:
            return {"available": False, "reason": str(exc)}

        high_pairs = []
        for i, col1 in enumerate(numerical_cols):
            for j, col2 in enumerate(numerical_cols):
                if j <= i:
                    continue
                val = corr_matrix.loc[col1, col2]
                if not math.isnan(float(val)) and abs(float(val)) >= 0.7:
                    high_pairs.append({
                        "col1": col1,
                        "col2": col2,
                        "correlation": _safe_float(val),
                        "strength": "very_high" if abs(float(val)) >= CORRELATION_LEAKAGE_THRESHOLD else "high",
                    })
        high_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        corr_dict: dict[str, dict[str, float | None]] = {}
        for col in numerical_cols:
            corr_dict[col] = {
                other: _safe_float(corr_matrix.loc[col, other])
                for other in numerical_cols
            }

        return {
            "available": True,
            "matrix": corr_dict,
            "highly_correlated_pairs": high_pairs[:20],
            "num_high_pairs": len(high_pairs),
        }

    @staticmethod
    def _outlier_summary(df: pd.DataFrame, metadata: dict) -> dict:
        numerical_cols = [
            c for c in metadata.get("numerical_columns", [])
            if c in df.columns
        ] or [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
        ]
        per_column: dict[str, Any] = {}

        for col in numerical_cols:
            series = df[col].dropna()
            if len(series) < 10:
                continue

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - OUTLIER_IQR_MULTIPLIER * iqr
                upper = q3 + OUTLIER_IQR_MULTIPLIER * iqr
                iqr_outliers = int(((series < lower) | (series > upper)).sum())
            else:
                iqr_outliers = 0
                lower = upper = float(series.iloc[0])

            mean = float(series.mean())
            std = float(series.std())
            if std > 0:
                z_scores = ((series - mean) / std).abs()
                zscore_outliers = int((z_scores > OUTLIER_ZSCORE_THRESHOLD).sum())
            else:
                zscore_outliers = 0

            per_column[col] = {
                "iqr_outliers": iqr_outliers,
                "iqr_outlier_pct": _safe_float(iqr_outliers / len(series) * 100),
                "iqr_lower_bound": _safe_float(lower),
                "iqr_upper_bound": _safe_float(upper),
                "zscore_outliers": zscore_outliers,
                "zscore_outlier_pct": _safe_float(zscore_outliers / len(series) * 100),
            }

        total_iqr = sum(v["iqr_outliers"] for v in per_column.values())
        columns_with_outliers = sum(1 for v in per_column.values() if v["iqr_outliers"] > 0)

        return {
            "total_iqr_outliers": total_iqr,
            "columns_with_outliers": columns_with_outliers,
            "per_column": per_column,
        }

    @staticmethod
    def _target_analysis(df: pd.DataFrame, target_column: str, metadata: dict) -> dict:
        return compute_target_analysis(df, target_column=target_column, metadata=metadata)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(value) -> float | None:
    """Convert to Python float; return None for NaN/Inf."""
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 6)
    except (TypeError, ValueError):
        return None


def _is_approximately_normal(series: pd.Series, threshold: float = 1.0) -> bool:
    """Rough normality check: |skewness| < threshold."""
    try:
        return abs(float(series.skew())) < threshold
    except Exception:
        return False

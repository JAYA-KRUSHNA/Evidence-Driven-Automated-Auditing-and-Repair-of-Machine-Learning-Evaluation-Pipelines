"""
EDA Engine — Automated Exploratory Data Analysis.

Produces a comprehensive analysis of any tabular dataset without
hard-coding any column names.

Analysis modules
----------------
    overview          – shape, dtypes, memory, file info
    statistics        – descriptive stats for numerical + categorical columns
    missing_analysis  – per-column missing value report
    distribution      – value counts, skewness, kurtosis, percentiles
    correlation       – correlation matrix + highly-correlated pairs
    outlier_summary   – IQR + Z-score based outlier counts per column
    target_analysis   – target distribution, class stats, feature importance

Public API
----------
    EDAEngine.analyze(df, metadata, target_column=None)
        → dict  (all analysis modules combined)
    EDAEngine.save(result, dataset_id, session=None)
        → EDAResult ORM row
"""

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from ..config import (
    OUTLIER_IQR_MULTIPLIER,
    OUTLIER_ZSCORE_THRESHOLD,
    CORRELATION_LEAKAGE_THRESHOLD,
)

logger = logging.getLogger(__name__)


class EDAEngine:
    """Run automated EDA on a DataFrame."""

    # ── Main entry point ──────────────────────────────────────────────────

    @staticmethod
    def analyze(
        df: pd.DataFrame,
        metadata: dict[str, Any],
        target_column: str | None = None,
    ) -> dict[str, Any]:
        """
        Run the full EDA pipeline and return a serialisable result dict.
        All values are JSON-safe (no numpy types, no NaN — replaced with None).
        """
        result: dict[str, Any] = {}

        result["overview"] = EDAEngine._overview(df, metadata)
        result["statistics"] = EDAEngine._statistics(df, metadata)
        result["missing_analysis"] = EDAEngine._missing_analysis(df)
        result["distribution"] = EDAEngine._distribution(df, metadata)
        result["correlation"] = EDAEngine._correlation(df, metadata)
        result["outlier_summary"] = EDAEngine._outlier_summary(df, metadata)

        if target_column and target_column in df.columns:
            result["target_analysis"] = EDAEngine._target_analysis(df, target_column, metadata)
        else:
            result["target_analysis"] = {"available": False, "reason": "No target column specified"}

        logger.info("EDA complete for dataset with shape %s", df.shape)
        return result

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
    #  Module 1 — Overview
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _overview(df: pd.DataFrame, metadata: dict) -> dict:
        num_rows, num_cols = df.shape
        return {
            "num_rows": num_rows,
            "num_columns": num_cols,
            "total_cells": num_rows * num_cols,
            "total_missing_cells": int(df.isnull().sum().sum()),
            "missing_pct": _safe_float(df.isnull().sum().sum() / (num_rows * num_cols) * 100),
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_bytes": int(df.memory_usage(deep=True).sum()),
            "memory_mb": _safe_float(df.memory_usage(deep=True).sum() / 1_048_576),
            "numerical_columns": metadata.get("numerical_columns", []),
            "categorical_columns": metadata.get("categorical_columns", []),
            "datetime_columns": metadata.get("datetime_columns", []),
            "identifier_columns": metadata.get("identifier_columns", []),
            "column_types": metadata.get("column_types", {}),
            "file_format": metadata.get("file_format"),
            "file_size_bytes": metadata.get("file_size_bytes"),
        }

    # ═══════════════════════════════════════════════════════════════════════
    #  Module 2 — Descriptive statistics
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _statistics(df: pd.DataFrame, metadata: dict) -> dict:
        numerical_cols = [c for c in metadata.get("numerical_columns", []) if c in df.columns]
        categorical_cols = [c for c in metadata.get("categorical_columns", []) if c in df.columns]

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

    # ═══════════════════════════════════════════════════════════════════════
    #  Module 3 — Missing value analysis
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _missing_analysis(df: pd.DataFrame) -> dict:
        num_rows = len(df)
        per_column: dict[str, Any] = {}
        for col in df.columns:
            missing_count = int(df[col].isnull().sum())
            if missing_count > 0:
                per_column[col] = {
                    "missing_count": missing_count,
                    "missing_pct": _safe_float(missing_count / num_rows * 100),
                    "dtype": str(df[col].dtype),
                }

        # Rows with at least one missing value
        rows_with_missing = int(df.isnull().any(axis=1).sum())

        return {
            "total_missing_cells": int(df.isnull().sum().sum()),
            "columns_with_missing": len(per_column),
            "rows_with_missing": rows_with_missing,
            "rows_with_missing_pct": _safe_float(rows_with_missing / num_rows * 100),
            "per_column": per_column,
        }

    # ═══════════════════════════════════════════════════════════════════════
    #  Module 4 — Distribution analysis
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _distribution(df: pd.DataFrame, metadata: dict) -> dict:
        numerical_cols = [c for c in metadata.get("numerical_columns", []) if c in df.columns]
        categorical_cols = [c for c in metadata.get("categorical_columns", []) if c in df.columns]

        numerical_dist: dict[str, Any] = {}
        for col in numerical_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            # 10-bin histogram
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

    # ═══════════════════════════════════════════════════════════════════════
    #  Module 5 — Correlation analysis
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _correlation(df: pd.DataFrame, metadata: dict) -> dict:
        numerical_cols = [c for c in metadata.get("numerical_columns", []) if c in df.columns]

        if len(numerical_cols) < 2:
            return {"available": False, "reason": "Need at least 2 numerical columns"}

        num_df = df[numerical_cols].dropna()
        if len(num_df) < 5:
            return {"available": False, "reason": "Insufficient rows after dropping NaN"}

        try:
            corr_matrix = num_df.corr()
        except Exception as exc:
            return {"available": False, "reason": str(exc)}

        # Find highly correlated pairs
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

        # Serialise the full matrix
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

    # ═══════════════════════════════════════════════════════════════════════
    #  Module 6 — Outlier summary
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _outlier_summary(df: pd.DataFrame, metadata: dict) -> dict:
        numerical_cols = [c for c in metadata.get("numerical_columns", []) if c in df.columns]
        per_column: dict[str, Any] = {}

        for col in numerical_cols:
            series = df[col].dropna()
            if len(series) < 10:
                continue

            # IQR method
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

            # Z-score method
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

    # ═══════════════════════════════════════════════════════════════════════
    #  Module 7 — Target analysis
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _target_analysis(
        df: pd.DataFrame,
        target_column: str,
        metadata: dict,
    ) -> dict:
        target = df[target_column].dropna()
        numerical_cols = [
            c for c in metadata.get("numerical_columns", [])
            if c in df.columns and c != target_column
        ]

        # Is it categorical or continuous?
        nunique = int(target.nunique())
        is_classification = (
            nunique <= 25
            or target.dtype == object
            or str(target.dtype) in ("category", "bool")
        )

        result: dict[str, Any] = {
            "available": True,
            "target_column": target_column,
            "num_rows": int(len(target)),
            "missing": int(df[target_column].isnull().sum()),
            "nunique": nunique,
            "inferred_task": "classification" if is_classification else "regression",
        }

        if is_classification:
            vc = target.value_counts()
            class_dist = {str(k): int(v) for k, v in vc.items()}
            class_pct = {str(k): _safe_float(v / len(target) * 100) for k, v in vc.items()}
            majority = int(vc.iloc[0])
            minority = int(vc.iloc[-1])
            ratio = _safe_float(majority / max(minority, 1))

            result.update({
                "class_distribution": class_dist,
                "class_pct": class_pct,
                "num_classes": len(vc),
                "majority_class": str(vc.index[0]),
                "minority_class": str(vc.index[-1]),
                "imbalance_ratio": ratio,
                "is_balanced": ratio <= 3.0,
            })
        else:
            result.update({
                "mean": _safe_float(target.mean()),
                "std": _safe_float(target.std()),
                "min": _safe_float(target.min()),
                "max": _safe_float(target.max()),
                "median": _safe_float(target.median()),
                "skewness": _safe_float(target.skew()),
            })

        # Feature-target correlation (numerical only)
        if numerical_cols:
            correlations: dict[str, float | None] = {}
            for col in numerical_cols:
                try:
                    if is_classification and target.dtype not in (np.float64, np.int64):
                        # skip if target is categorical text
                        continue
                    corr = _safe_float(df[col].corr(df[target_column]))
                    correlations[col] = corr
                except Exception:
                    correlations[col] = None

            # Top 10 by absolute value
            top_corr = sorted(
                [(k, v) for k, v in correlations.items() if v is not None],
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:10]
            result["feature_target_correlation"] = {k: v for k, v in top_corr}

        return result


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

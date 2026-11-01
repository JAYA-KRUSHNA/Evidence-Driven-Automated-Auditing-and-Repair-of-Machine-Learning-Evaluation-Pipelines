"""
EDA Module — Target Variable Analysis Engine.

Performs deep analysis of the target feature:
- Automatic target column suggestions and manual selection
- Automatic task inference with manual task_type override (classification vs regression)
- Classification target: Class distribution, class imbalance ratio, minority class %,
  ANOVA F-scores / Chi-square relationships with features, mutual information.
- Regression target: Descriptive stats, skewness, kurtosis, target outliers,
  Pearson & Spearman feature-target correlations, mutual information regression.
"""

from typing import Any
import math
import re
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def _safe_float(val: Any) -> float | None:
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 6)
    except (TypeError, ValueError):
        return None


TARGET_KEYWORDS = [
    "target", "label", "class", "outcome", "status", "churn",
    "survived", "price", "salary", "sales", "revenue", "cost", "score", "risk"
]


def suggest_target_columns(df: pd.DataFrame) -> list[str]:
    """Suggest likely target columns based on column names and positions."""
    suggestions: list[str] = []
    cols = list(df.columns)

    # 1. Name heuristics
    for col in cols:
        col_lower = col.lower()
        if any(kw == col_lower or kw in col_lower for kw in TARGET_KEYWORDS):
            suggestions.append(col)

    # 2. Add last column if not already suggested
    if cols and cols[-1] not in suggestions:
        suggestions.append(cols[-1])

    # 3. Add any other column that is binary
    for col in cols:
        if col not in suggestions and df[col].nunique() == 2:
            suggestions.append(col)

    return suggestions[:5]


def compute_target_analysis(
    df: pd.DataFrame,
    target_column: str | None = None,
    task_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run complete target variable analysis with automatic task detection and override support.
    Maintains all legacy keys for backwards compatibility.
    """
    if metadata is None:
        metadata = {}

    suggestions = suggest_target_columns(df)

    if not target_column or target_column not in df.columns:
        return {
            "available": False,
            "reason": "No target column specified",
            "suggested_targets": suggestions,
        }

    target = df[target_column].dropna()
    num_rows = int(len(target))
    missing_cnt = int(df[target_column].isnull().sum())
    nunique = int(target.nunique())

    # Task type detection / override
    if task_type in ("classification", "regression"):
        inferred_task = task_type
    else:
        is_classification = (
            nunique <= 25
            or target.dtype == object
            or str(target.dtype) in ("category", "bool")
        )
        inferred_task = "classification" if is_classification else "regression"

    numerical_cols = [
        c for c in df.columns
        if c != target_column and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]
    categorical_cols = [
        c for c in df.columns
        if c != target_column and c not in numerical_cols
    ]

    result: dict[str, Any] = {
        # Legacy base keys
        "available": True,
        "target_column": target_column,
        "num_rows": num_rows,
        "missing": missing_cnt,
        "nunique": nunique,
        "inferred_task": inferred_task,
        "suggested_targets": suggestions,
    }

    if inferred_task == "classification":
        vc = target.value_counts()
        class_dist = {str(k): int(v) for k, v in vc.items()}
        class_pct = {str(k): _safe_float((v / max(num_rows, 1)) * 100.0) for k, v in vc.items()}
        majority_val = int(vc.iloc[0]) if len(vc) > 0 else 0
        minority_val = int(vc.iloc[-1]) if len(vc) > 0 else 0
        ratio = _safe_float(majority_val / max(minority_val, 1))

        imbalance_status = (
            "Severe Imbalance (ratio > 10)" if (ratio or 1) > 10 else
            "Moderate Imbalance (ratio > 3)" if (ratio or 1) > 3 else
            "Balanced"
        )

        result.update({
            "class_distribution": class_dist,
            "class_pct": class_pct,
            "num_classes": len(vc),
            "majority_class": str(vc.index[0]) if len(vc) > 0 else None,
            "minority_class": str(vc.index[-1]) if len(vc) > 0 else None,
            "imbalance_ratio": ratio,
            "is_balanced": (ratio or 1) <= 3.0,
            "imbalance_status": imbalance_status,
            "minority_percentage": class_pct.get(str(vc.index[-1])) if len(vc) > 0 else None,
        })

        # Feature-Target associations: ANOVA F-test for numeric features
        feature_relationships = []
        if numerical_cols and len(vc) >= 2:
            for col in numerical_cols:
                col_data = df[[col, target_column]].dropna()
                groups = [grp[col].values for _, grp in col_data.groupby(target_column) if len(grp) > 0]
                if len(groups) >= 2:
                    try:
                        f_stat, p_val = stats.f_oneway(*groups)
                        feature_relationships.append({
                            "feature": col,
                            "type": "numerical",
                            "test": "ANOVA F-test",
                            "statistic": _safe_float(f_stat),
                            "p_value": _safe_float(p_val),
                            "significant": bool(p_val < 0.05),
                        })
                    except Exception:
                        pass
        feature_relationships.sort(key=lambda x: x["statistic"] or 0, reverse=True)
        result["feature_relationships"] = feature_relationships

        # Legacy feature_target_correlation (for numeric targets/columns)
        correlations: dict[str, float | None] = {}
        if pd.api.types.is_numeric_dtype(target):
            for col in numerical_cols:
                try:
                    correlations[col] = _safe_float(df[col].corr(df[target_column]))
                except Exception:
                    correlations[col] = None
        top_corr = sorted(
            [(k, v) for k, v in correlations.items() if v is not None],
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:10]
        result["feature_target_correlation"] = {k: v for k, v in top_corr}

    else:
        # Regression target
        mean_val = _safe_float(target.mean())
        std_val = _safe_float(target.std())
        median_val = _safe_float(target.median())
        min_val = _safe_float(target.min())
        max_val = _safe_float(target.max())
        skew_val = _safe_float(target.skew())
        kurt_val = _safe_float(target.kurtosis())

        q1 = target.quantile(0.25)
        q3 = target.quantile(0.75)
        iqr = q3 - q1
        outliers_cnt = int(((target < q1 - 1.5 * iqr) | (target > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0

        result.update({
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "median": median_val,
            "skewness": skew_val,
            "kurtosis": kurt_val,
            "iqr": _safe_float(iqr),
            "target_outliers_count": outliers_cnt,
            "target_outliers_pct": _safe_float((outliers_cnt / max(num_rows, 1)) * 100.0),
        })

        # Feature-Target correlations
        correlations = {}
        spearman_correlations = {}
        if numerical_cols:
            for col in numerical_cols:
                sub_corr = df[[col, target_column]].dropna()
                if len(sub_corr) >= 3:
                    try:
                        p_val, _ = stats.pearsonr(sub_corr[col], sub_corr[target_column])
                        correlations[col] = _safe_float(p_val)
                    except Exception:
                        correlations[col] = None
                    try:
                        s_val, _ = stats.spearmanr(sub_corr[col], sub_corr[target_column])
                        spearman_correlations[col] = _safe_float(s_val)
                    except Exception:
                        spearman_correlations[col] = None

        top_corr = sorted(
            [(k, v) for k, v in correlations.items() if v is not None],
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:10]
        result["feature_target_correlation"] = {k: v for k, v in top_corr}
        result["feature_target_spearman"] = {
            k: spearman_correlations[k] for k, _ in top_corr if k in spearman_correlations
        }

    return result

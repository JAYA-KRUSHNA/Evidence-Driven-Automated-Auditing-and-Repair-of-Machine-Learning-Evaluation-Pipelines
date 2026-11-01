"""
EDA Module — Automated Feature Analysis & Selection Engine.

Audits feature matrix to identify:
- Highly correlated feature pairs (|r| >= 0.8)
- Redundant / duplicate features
- Low-variance / near-constant features (variance < 0.01)
- High-cardinality categorical features
- Potential target data leakage (|r| >= 0.95 with target)
- Irrelevant features (zero predictive association)
- Multicollinear feature clusters
Generates prioritized feature selection recommendations without deleting any features.
"""

from typing import Any
import math
import numpy as np
import pandas as pd
from scipy import stats

from ..config import CORRELATION_LEAKAGE_THRESHOLD


def _safe_float(val: Any) -> float | None:
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 6)
    except (TypeError, ValueError):
        return None


def compute_feature_analysis(
    df: pd.DataFrame,
    target_column: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run automated feature analysis and return identified anomalies and recommendations.
    """
    if metadata is None:
        metadata = {}

    num_rows = len(df)
    feature_cols = [c for c in df.columns if c != target_column]

    numerical_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]
    categorical_cols = [
        c for c in feature_cols
        if c not in numerical_cols
    ]

    recommendations: list[dict[str, Any]] = []

    # 1. Highly Correlated Pairs (|r| >= 0.80)
    high_corr_pairs: list[dict[str, Any]] = []
    if len(numerical_cols) >= 2:
        num_df = df[numerical_cols].dropna()
        if len(num_df) >= 5:
            corr_mat = num_df.corr().abs()
            for i in range(len(numerical_cols)):
                for j in range(i + 1, len(numerical_cols)):
                    c1, c2 = numerical_cols[i], numerical_cols[j]
                    val = float(corr_mat.loc[c1, c2])
                    if not math.isnan(val) and val >= 0.80:
                        high_corr_pairs.append({
                            "feature1": c1,
                            "feature2": c2,
                            "correlation": _safe_float(val),
                            "severity": "critical" if val >= 0.95 else "warning",
                        })
                        recommendations.append({
                            "type": "drop_correlated_feature",
                            "features": [c1, c2],
                            "recommendation": f"Features '{c1}' and '{c2}' are {val:.2f} correlated. Drop one to prevent multicollinearity in linear models.",
                            "priority": "high" if val >= 0.95 else "medium",
                        })
    high_corr_pairs.sort(key=lambda x: x["correlation"] or 0, reverse=True)

    # 2. Low-Variance Features
    low_variance_features: list[dict[str, Any]] = []
    for col in numerical_cols:
        clean = df[col].dropna()
        if len(clean) >= 2:
            var_val = float(clean.var())
            if var_val < 0.01:
                low_variance_features.append({
                    "feature": col,
                    "variance": _safe_float(var_val),
                    "std": _safe_float(clean.std()),
                })
                recommendations.append({
                    "type": "low_variance",
                    "features": [col],
                    "recommendation": f"Feature '{col}' has near-zero variance ({var_val:.5f}). Consider dropping it to reduce dimensionality.",
                    "priority": "medium",
                })

    # 3. High-Cardinality Categoricals
    high_cardinality_features: list[dict[str, Any]] = []
    for col in categorical_cols:
        clean = df[col].dropna()
        if len(clean) >= 20:
            nunique = int(clean.nunique())
            ratio = nunique / len(clean)
            if nunique > 50 and ratio > 0.2:
                high_cardinality_features.append({
                    "feature": col,
                    "nunique": nunique,
                    "ratio": _safe_float(ratio * 100.0),
                })
                recommendations.append({
                    "type": "high_cardinality",
                    "features": [col],
                    "recommendation": f"Categorical feature '{col}' has {nunique} unique levels. Use Target Encoding or group rare categories before tree/linear modeling.",
                    "priority": "medium",
                })

    # 4. Potential Data Leakage with Target
    leakage_features: list[dict[str, Any]] = []
    if target_column and target_column in df.columns:
        target_ser = df[target_column].dropna()
        is_num_target = pd.api.types.is_numeric_dtype(target_ser)

        if is_num_target:
            for col in numerical_cols:
                sub = df[[col, target_column]].dropna()
                if len(sub) >= 5:
                    try:
                        r = abs(float(sub[col].corr(sub[target_column])))
                        if r >= CORRELATION_LEAKAGE_THRESHOLD:
                            leakage_features.append({
                                "feature": col,
                                "correlation_with_target": _safe_float(r),
                                "risk": "High Leakage Risk",
                            })
                            recommendations.append({
                                "type": "target_leakage",
                                "features": [col],
                                "recommendation": f"Feature '{col}' has extreme correlation ({r:.3f}) with target '{target_column}'. Verify it is not a direct proxy or recorded post-event (target leakage).",
                                "priority": "critical",
                            })
                    except Exception:
                        pass

    # 5. Potentially Irrelevant Features (Near-zero predictive signal)
    irrelevant_features: list[dict[str, Any]] = []
    if target_column and target_column in df.columns and len(numerical_cols) >= 3:
        target_ser = df[target_column].dropna()
        if pd.api.types.is_numeric_dtype(target_ser):
            for col in numerical_cols:
                sub = df[[col, target_column]].dropna()
                if len(sub) >= 10:
                    try:
                        r = abs(float(sub[col].corr(sub[target_column])))
                        if r < 0.01:
                            irrelevant_features.append({
                                "feature": col,
                                "correlation_with_target": _safe_float(r),
                            })
                    except Exception:
                        pass

    return {
        "features_evaluated": len(feature_cols),
        "numerical_features_count": len(numerical_cols),
        "categorical_features_count": len(categorical_cols),
        "highly_correlated_pairs": high_corr_pairs,
        "low_variance_features": low_variance_features,
        "high_cardinality_features": high_cardinality_features,
        "potential_leakage_features": leakage_features,
        "irrelevant_features": irrelevant_features,
        "feature_selection_recommendations": recommendations,
    }

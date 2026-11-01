"""
EDA Module — Advanced Multi-Method Outlier Detection Engine.

Implements five complementary outlier detection algorithms:
1. Interquartile Range (IQR / Tukey's Fences)
2. Standard Z-Score (|Z| > 3.0)
3. Modified Z-Score (Median Absolute Deviation - MAD > 3.5)
4. Isolation Forest (Unsupervised tree ensemble anomaly detector)
5. Local Outlier Factor (LOF - Density-based local neighborhood anomaly detector)

Calculates counts, percentages, affected columns, sample anomalies,
and actionable treatment recommendations (Capping, Robust Scaling, Imputation).
NEVER deletes records automatically.
"""

from typing import Any
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from ..config import OUTLIER_IQR_MULTIPLIER, OUTLIER_ZSCORE_THRESHOLD


def _safe_float(val: Any) -> float | None:
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 6)
    except (TypeError, ValueError):
        return None


def detect_iqr_outliers(series: pd.Series, multiplier: float = OUTLIER_IQR_MULTIPLIER) -> dict[str, Any]:
    """Detect outliers using IQR Tukey's fences."""
    clean = series.dropna()
    n = len(clean)
    if n < 5:
        return {"count": 0, "pct": 0.0, "lower": None, "upper": None, "indices": []}

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1

    if iqr <= 0:
        return {"count": 0, "pct": 0.0, "lower": _safe_float(q1), "upper": _safe_float(q3), "indices": []}

    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    mask = (clean < lower) | (clean > upper)
    outlier_indices = clean[mask].index.tolist()
    cnt = len(outlier_indices)
    pct = (cnt / n) * 100.0

    return {
        "count": cnt,
        "pct": _safe_float(pct),
        "lower": _safe_float(lower),
        "upper": _safe_float(upper),
        "indices": outlier_indices[:50],
        "sample_values": [_safe_float(v) for v in clean[mask].head(10).tolist()],
    }


def detect_zscore_outliers(series: pd.Series, threshold: float = OUTLIER_ZSCORE_THRESHOLD) -> dict[str, Any]:
    """Detect outliers using standard Z-score (|z| > threshold)."""
    clean = series.dropna()
    n = len(clean)
    if n < 5:
        return {"count": 0, "pct": 0.0, "threshold": threshold, "indices": []}

    mean = float(clean.mean())
    std = float(clean.std())

    if std <= 1e-9:
        return {"count": 0, "pct": 0.0, "threshold": threshold, "indices": []}

    z_scores = ((clean - mean) / std).abs()
    mask = z_scores > threshold
    outlier_indices = clean[mask].index.tolist()
    cnt = len(outlier_indices)
    pct = (cnt / n) * 100.0

    return {
        "count": cnt,
        "pct": _safe_float(pct),
        "threshold": threshold,
        "indices": outlier_indices[:50],
        "sample_values": [_safe_float(v) for v in clean[mask].head(10).tolist()],
    }


def detect_modified_zscore_outliers(series: pd.Series, threshold: float = 3.5) -> dict[str, Any]:
    """
    Detect outliers using Modified Z-score (Boris Iglewicz and David Hoaglin).
    Uses Median Absolute Deviation (MAD), robust against masking effects.
    """
    clean = series.dropna()
    n = len(clean)
    if n < 5:
        return {"count": 0, "pct": 0.0, "threshold": threshold, "indices": []}

    median = float(clean.median())
    mad = float((clean - median).abs().median())

    if mad <= 1e-9:
        return {"count": 0, "pct": 0.0, "threshold": threshold, "indices": []}

    mod_z = 0.6745 * (clean - median).abs() / mad
    mask = mod_z > threshold
    outlier_indices = clean[mask].index.tolist()
    cnt = len(outlier_indices)
    pct = (cnt / n) * 100.0

    return {
        "count": cnt,
        "pct": _safe_float(pct),
        "threshold": threshold,
        "indices": outlier_indices[:50],
        "sample_values": [_safe_float(v) for v in clean[mask].head(10).tolist()],
    }


def detect_isolation_forest_outliers(df: pd.DataFrame, numerical_cols: list[str], contamination: float = 0.05) -> dict[str, Any]:
    """Detect multivariate anomalies using Isolation Forest."""
    sub = df[numerical_cols].dropna()
    n = len(sub)
    if n < 10 or len(numerical_cols) == 0:
        return {"count": 0, "pct": 0.0, "indices": []}

    # Cap rows if extremely large for fast interactive execution
    sample_sub = sub if n <= 3000 else sub.sample(3000, random_state=42)

    try:
        iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        preds = iso.fit_predict(sample_sub)
        outlier_mask = preds == -1
        outlier_indices = sample_sub[outlier_mask].index.tolist()
        cnt = len(outlier_indices)
        pct = (cnt / len(sample_sub)) * 100.0

        return {
            "count": cnt,
            "pct": _safe_float(pct),
            "contamination_parameter": contamination,
            "indices": outlier_indices[:50],
        }
    except Exception:
        return {"count": 0, "pct": 0.0, "indices": []}


def detect_lof_outliers(df: pd.DataFrame, numerical_cols: list[str], n_neighbors: int = 20) -> dict[str, Any]:
    """Detect local density anomalies using Local Outlier Factor (LOF)."""
    sub = df[numerical_cols].dropna()
    n = len(sub)
    if n < 20 or len(numerical_cols) == 0:
        return {"count": 0, "pct": 0.0, "indices": []}

    sample_sub = sub if n <= 2000 else sub.sample(2000, random_state=42)
    k = min(n_neighbors, len(sample_sub) - 1)

    try:
        lof = LocalOutlierFactor(n_neighbors=k, contamination=0.05, n_jobs=-1)
        preds = lof.fit_predict(sample_sub)
        outlier_mask = preds == -1
        outlier_indices = sample_sub[outlier_mask].index.tolist()
        cnt = len(outlier_indices)
        pct = (cnt / len(sample_sub)) * 100.0

        return {
            "count": cnt,
            "pct": _safe_float(pct),
            "n_neighbors": k,
            "indices": outlier_indices[:50],
        }
    except Exception:
        return {"count": 0, "pct": 0.0, "indices": []}


def get_recommended_treatment(pct: float, method_name: str, col_name: str | None = None) -> str:
    """Provide a domain-grounded recommendation for handling outliers."""
    target_str = f"for column '{col_name}'" if col_name else "across numerical features"
    if pct == 0.0:
        return f"No action needed {target_str}. No significant anomalies detected."
    elif pct < 1.0:
        return f"Mild anomaly frequency ({pct:.1f}%). Retain original values or apply RobustScaler to prevent gradient distortion."
    elif pct <= 5.0:
        return f"Moderate outliers ({pct:.1f}%). Apply Winsorization (capping at 1st and 99th percentiles) in preprocessing to preserve sample size while bounding extreme influences."
    else:
        return f"High anomaly concentration ({pct:.1f}%). Investigate potential sensor/data collection artifacts; consider log or Box-Cox power transform."


def compute_outliers(
    df: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
    method: str = "all",
) -> dict[str, Any]:
    """
    Compute comprehensive outlier detection across all methods or a selected method.
    Maintains compatibility with legacy outlier_summary structure.
    """
    if metadata is None:
        metadata = {}

    numerical_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]

    iqr_per_col: dict[str, Any] = {}
    zscore_per_col: dict[str, Any] = {}
    mod_zscore_per_col: dict[str, Any] = {}

    total_iqr_outliers = 0
    total_zscore_outliers = 0
    total_mod_zscore_outliers = 0

    for col in numerical_cols:
        s = df[col]
        # IQR
        res_iqr = detect_iqr_outliers(s)
        iqr_per_col[col] = res_iqr
        total_iqr_outliers += res_iqr["count"]

        # Z-Score
        res_z = detect_zscore_outliers(s)
        zscore_per_col[col] = res_z
        total_zscore_outliers += res_z["count"]

        # Modified Z-Score
        res_mod = detect_modified_zscore_outliers(s)
        mod_zscore_per_col[col] = res_mod
        total_mod_zscore_outliers += res_mod["count"]

    # Legacy per_column structure for backwards compatibility
    legacy_per_column: dict[str, Any] = {}
    for col in numerical_cols:
        iqr_res = iqr_per_col[col]
        z_res = zscore_per_col[col]
        legacy_per_column[col] = {
            "iqr_outliers": iqr_res["count"],
            "iqr_outlier_pct": iqr_res["pct"],
            "iqr_lower_bound": iqr_res["lower"],
            "iqr_upper_bound": iqr_res["upper"],
            "zscore_outliers": z_res["count"],
            "zscore_outlier_pct": z_res["pct"],
        }

    columns_with_outliers = sum(1 for v in legacy_per_column.values() if v["iqr_outliers"] > 0)

    # Multivariate anomaly detectors
    iso_res = detect_isolation_forest_outliers(df, numerical_cols) if method in ("all", "isolation_forest") else {"count": 0, "pct": 0.0, "indices": []}
    lof_res = detect_lof_outliers(df, numerical_cols) if method in ("all", "lof") else {"count": 0, "pct": 0.0, "indices": []}

    # Summary table across methods
    num_rows = len(df)
    methods_comparison = [
        {
            "method": "IQR (Tukey's Fences)",
            "outlier_count": total_iqr_outliers,
            "outlier_pct": _safe_float((total_iqr_outliers / max(num_rows * max(len(numerical_cols), 1), 1)) * 100.0),
            "affected_columns": sum(1 for c in iqr_per_col.values() if c["count"] > 0),
            "recommendation": get_recommended_treatment(
                (total_iqr_outliers / max(num_rows * max(len(numerical_cols), 1), 1)) * 100.0,
                "IQR",
            ),
        },
        {
            "method": "Standard Z-Score (|Z| > 3)",
            "outlier_count": total_zscore_outliers,
            "outlier_pct": _safe_float((total_zscore_outliers / max(num_rows * max(len(numerical_cols), 1), 1)) * 100.0),
            "affected_columns": sum(1 for c in zscore_per_col.values() if c["count"] > 0),
            "recommendation": get_recommended_treatment(
                (total_zscore_outliers / max(num_rows * max(len(numerical_cols), 1), 1)) * 100.0,
                "Z-Score",
            ),
        },
        {
            "method": "Modified Z-Score (MAD > 3.5)",
            "outlier_count": total_mod_zscore_outliers,
            "outlier_pct": _safe_float((total_mod_zscore_outliers / max(num_rows * max(len(numerical_cols), 1), 1)) * 100.0),
            "affected_columns": sum(1 for c in mod_zscore_per_col.values() if c["count"] > 0),
            "recommendation": get_recommended_treatment(
                (total_mod_zscore_outliers / max(num_rows * max(len(numerical_cols), 1), 1)) * 100.0,
                "Modified Z-Score",
            ),
        },
        {
            "method": "Isolation Forest (Ensemble)",
            "outlier_count": iso_res["count"],
            "outlier_pct": iso_res["pct"],
            "affected_columns": len(numerical_cols) if iso_res["count"] > 0 else 0,
            "recommendation": get_recommended_treatment(iso_res["pct"] or 0.0, "Isolation Forest"),
        },
        {
            "method": "Local Outlier Factor (LOF)",
            "outlier_count": lof_res["count"],
            "outlier_pct": lof_res["pct"],
            "affected_columns": len(numerical_cols) if lof_res["count"] > 0 else 0,
            "recommendation": get_recommended_treatment(lof_res["pct"] or 0.0, "LOF"),
        },
    ]

    return {
        # Legacy keys
        "total_iqr_outliers": total_iqr_outliers,
        "columns_with_outliers": columns_with_outliers,
        "per_column": legacy_per_column,

        # Enhanced keys
        "methods_comparison": methods_comparison,
        "details_by_method": {
            "iqr": iqr_per_col,
            "zscore": zscore_per_col,
            "modified_zscore": mod_zscore_per_col,
            "isolation_forest": iso_res,
            "local_outlier_factor": lof_res,
        },
        "available_methods": ["iqr", "zscore", "modified_zscore", "isolation_forest", "lof"],
    }

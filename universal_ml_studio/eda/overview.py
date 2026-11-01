"""
EDA Module — Dataset Overview & Health Score Calculator.

Computes comprehensive dataset dimensions, column type classifications,
memory consumption, duplicate rows/columns, unique values, and an
automated 0-100 Dataset Health Score.
"""

from typing import Any
import math
import numpy as np
import pandas as pd


def _safe_float(val: Any) -> float | None:
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 6)
    except (TypeError, ValueError):
        return None


def calculate_health_score(
    num_rows: int,
    num_cols: int,
    total_missing: int,
    duplicate_rows: int,
    constant_cols_count: int,
    duplicate_cols_count: int,
    outlier_pct: float,
    high_cardinality_count: int,
) -> dict[str, Any]:
    """
    Calculate an intelligent 0-100 Dataset Health Score with itemized penalties.
    """
    total_cells = max(num_rows * num_cols, 1)
    missing_pct = (total_missing / total_cells) * 100.0
    duplicate_pct = (duplicate_rows / max(num_rows, 1)) * 100.0

    score = 100.0
    penalties = []

    # 1. Missing values penalty (up to 30 pts)
    if missing_pct > 0:
        missing_pen = min(30.0, (missing_pct / 50.0) * 30.0)
        score -= missing_pen
        penalties.append({
            "dimension": "Missing Data",
            "penalty": round(missing_pen, 1),
            "reason": f"{missing_pct:.1f}% of all cells are missing values",
            "severity": "high" if missing_pct > 15 else "medium" if missing_pct > 5 else "low",
        })

    # 2. Duplicate rows penalty (up to 20 pts)
    if duplicate_pct > 0:
        dup_pen = min(20.0, (duplicate_pct / 20.0) * 20.0)
        score -= dup_pen
        penalties.append({
            "dimension": "Duplicate Records",
            "penalty": round(dup_pen, 1),
            "reason": f"{duplicate_pct:.1f}% of rows are exact duplicates ({duplicate_rows} rows)",
            "severity": "high" if duplicate_pct > 10 else "medium",
        })

    # 3. Constant and duplicate columns penalty (up to 15 pts)
    useless_cols = constant_cols_count + duplicate_cols_count
    if useless_cols > 0:
        col_pen = min(15.0, (useless_cols / max(num_cols, 1)) * 30.0)
        score -= col_pen
        penalties.append({
            "dimension": "Zero-Variance / Redundant Columns",
            "penalty": round(col_pen, 1),
            "reason": f"{constant_cols_count} constant and {duplicate_cols_count} duplicate columns detected",
            "severity": "medium",
        })

    # 4. Outliers penalty (up to 15 pts)
    if outlier_pct > 3.0:
        outlier_pen = min(15.0, ((outlier_pct - 3.0) / 20.0) * 15.0)
        score -= outlier_pen
        penalties.append({
            "dimension": "Outlier Extremes",
            "penalty": round(outlier_pen, 1),
            "reason": f"Approx {outlier_pct:.1f}% potential extreme values detected across numeric features",
            "severity": "medium" if outlier_pct > 10 else "low",
        })

    # 5. High cardinality penalty (up to 10 pts)
    if high_cardinality_count > 0:
        card_pen = min(10.0, high_cardinality_count * 2.5)
        score -= card_pen
        penalties.append({
            "dimension": "High Cardinality",
            "penalty": round(card_pen, 1),
            "reason": f"{high_cardinality_count} categorical columns have extremely high cardinality",
            "severity": "low",
        })

    final_score = max(0.0, min(100.0, round(score, 1)))

    if final_score >= 90:
        grade = "Excellent"
        status_color = "emerald"
    elif final_score >= 75:
        grade = "Good"
        status_color = "cyan"
    elif final_score >= 55:
        grade = "Fair"
        status_color = "amber"
    else:
        grade = "Needs Attention"
        status_color = "rose"

    return {
        "score": final_score,
        "grade": grade,
        "status_color": status_color,
        "penalties": penalties,
    }


def compute_overview(df: pd.DataFrame, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Compute comprehensive dataset overview metrics and health score.
    Maintains all legacy keys for backwards compatibility.
    """
    if metadata is None:
        metadata = {}

    num_rows, num_cols = df.shape
    total_cells = num_rows * num_cols
    total_missing = int(df.isnull().sum().sum())
    missing_pct = _safe_float((total_missing / max(total_cells, 1)) * 100.0)

    # Memory usage
    try:
        mem_bytes = int(df.memory_usage(deep=True).sum())
    except Exception:
        mem_bytes = int(df.memory_usage().sum())
    mem_kb = _safe_float(mem_bytes / 1024.0)
    mem_mb = _safe_float(mem_bytes / 1_048_576.0)

    # Duplicate rows
    dup_rows = int(df.duplicated().sum())
    dup_rows_pct = _safe_float((dup_rows / max(num_rows, 1)) * 100.0)

    # Column type classifications
    numerical_cols: list[str] = []
    categorical_cols: list[str] = []
    boolean_cols: list[str] = []
    datetime_cols: list[str] = []
    column_types: dict[str, str] = {}
    unique_counts: dict[str, int] = {}
    constant_cols: list[str] = []

    for col in df.columns:
        s = df[col]
        dtype_str = str(s.dtype)
        nunique = int(s.nunique(dropna=True))
        unique_counts[col] = nunique

        if nunique <= 1 and num_rows > 0:
            constant_cols.append(col)

        if pd.api.types.is_bool_dtype(s):
            boolean_cols.append(col)
            column_types[col] = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(s):
            datetime_cols.append(col)
            column_types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            numerical_cols.append(col)
            column_types[col] = "numerical"
        else:
            # Check if object can be parsed as datetime
            categorical_cols.append(col)
            column_types[col] = "categorical"

    # Merge with metadata if present
    meta_num = metadata.get("numerical_columns")
    if meta_num is not None:
        numerical_cols = [c for c in meta_num if c in df.columns]
    meta_cat = metadata.get("categorical_columns")
    if meta_cat is not None:
        categorical_cols = [c for c in meta_cat if c in df.columns]
    meta_dt = metadata.get("datetime_columns")
    if meta_dt is not None:
        datetime_cols = [c for c in meta_dt if c in df.columns]
    meta_types = metadata.get("column_types")
    if meta_types:
        column_types.update({k: v for k, v in meta_types.items() if k in df.columns})

    # Duplicate columns detection
    duplicate_cols: list[tuple[str, str]] = []
    sample_df = df if len(df) <= 1000 else df.sample(1000, random_state=42)
    cols_list = list(df.columns)
    for i in range(len(cols_list)):
        for j in range(i + 1, len(cols_list)):
            c1, c2 = cols_list[i], cols_list[j]
            try:
                if sample_df[c1].equals(sample_df[c2]):
                    duplicate_cols.append((c1, c2))
            except Exception:
                pass

    # Rough outlier percentage for health score calculation
    outlier_pct_est = 0.0
    if numerical_cols and num_rows >= 10:
        outlier_counts = 0
        for col in numerical_cols[:10]:  # sample up to 10 cols for speed
            ser = df[col].dropna()
            if len(ser) >= 10:
                q1 = ser.quantile(0.25)
                q3 = ser.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    cnt = ((ser < q1 - 1.5 * iqr) | (ser > q3 + 1.5 * iqr)).sum()
                    outlier_counts += cnt
        outlier_pct_est = (outlier_counts / max(num_rows * min(len(numerical_cols), 10), 1)) * 100.0

    # High cardinality count
    high_card_count = sum(1 for c in categorical_cols if unique_counts.get(c, 0) > 50 and num_rows > 100)

    # Health score
    health = calculate_health_score(
        num_rows=num_rows,
        num_cols=num_cols,
        total_missing=total_missing,
        duplicate_rows=dup_rows,
        constant_cols_count=len(constant_cols),
        duplicate_cols_count=len(duplicate_cols),
        outlier_pct=outlier_pct_est,
        high_cardinality_count=high_card_count,
    )

    return {
        # Legacy keys
        "num_rows": num_rows,
        "num_columns": num_cols,
        "total_cells": total_cells,
        "total_missing_cells": total_missing,
        "missing_pct": missing_pct,
        "duplicate_rows": dup_rows,
        "memory_bytes": mem_bytes,
        "memory_mb": mem_mb,
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "identifier_columns": metadata.get("identifier_columns", []),
        "column_types": column_types,
        "file_format": metadata.get("file_format", "csv"),
        "file_size_bytes": metadata.get("file_size_bytes", mem_bytes),

        # Enhanced keys
        "dimensions": f"{num_rows:,} rows × {num_cols:,} columns",
        "duplicate_rows_pct": dup_rows_pct,
        "duplicate_columns": duplicate_cols,
        "duplicate_columns_count": len(duplicate_cols),
        "constant_columns": constant_cols,
        "constant_columns_count": len(constant_cols),
        "boolean_columns": boolean_cols,
        "num_numerical": len(numerical_cols),
        "num_categorical": len(categorical_cols),
        "num_boolean": len(boolean_cols),
        "num_datetime": len(datetime_cols),
        "memory_kb": mem_kb,
        "unique_counts": unique_counts,
        "health_score": health["score"],
        "health_grade": health["grade"],
        "health_status_color": health["status_color"],
        "health_penalties": health["penalties"],
    }

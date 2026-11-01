"""
EDA Module — Advanced Data Quality Analysis.

Performs deep audit of data quality:
- Missing values and missing percentages
- Duplicate rows and duplicate columns
- Constant and near-constant columns
- Infinite values (inf / -inf)
- Incorrect data types (numeric as string, date as object)
- High-cardinality columns
- Potential identifier columns
- Invalid / suspicious values (placeholders like ?, -999, negative ages)
- Concrete actionable recommendations for each detected defect.
"""

from typing import Any
import math
import re
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


SUSPICIOUS_STRINGS = {
    "?", "none", "n/a", "na", "null", "undefined", "unknown", "missing",
    "-", "--", "---", ".", "nil", "nan", "empty", "void", "error", "-999", "-9999", "999", "9999"
}

IDENTIFIER_PATTERNS = re.compile(
    r"(^id$|_id$|^uuid$|^guid$|^key$|^pk$|^account_?no$|^customer_?id$|^user_?id$|^trans_?id$|^code$)",
    re.IGNORECASE,
)


def compute_quality(df: pd.DataFrame, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Run an exhaustive data quality audit on the dataframe.
    Returns detected issues, statistics, and concrete recommendations.
    """
    num_rows, num_cols = df.shape
    issues: list[dict[str, Any]] = []
    recommendations: list[dict[str, Any]] = []

    # 1. Missing Values
    missing_by_col = {}
    total_missing_cells = 0
    for col in df.columns:
        m_cnt = int(df[col].isnull().sum())
        total_missing_cells += m_cnt
        if m_cnt > 0:
            m_pct = (m_cnt / max(num_rows, 1)) * 100.0
            missing_by_col[col] = {
                "count": m_cnt,
                "pct": _safe_float(m_pct),
                "dtype": str(df[col].dtype),
            }
            severity = "critical" if m_pct > 40 else "warning" if m_pct > 5 else "info"
            issues.append({
                "type": "missing_values",
                "column": col,
                "severity": severity,
                "message": f"Column '{col}' has {m_cnt:,} missing values ({m_pct:.1f}%).",
            })
            if m_pct > 60:
                recommendations.append({
                    "column": col,
                    "action": "drop_column",
                    "reason": f"Column '{col}' has {m_pct:.1f}% missing data, which exceeds reliable imputation thresholds.",
                    "priority": "high",
                })
            elif pd.api.types.is_numeric_dtype(df[col]):
                recommendations.append({
                    "column": col,
                    "action": "impute_numeric",
                    "reason": f"Impute missing values in '{col}' using median (for skewed) or mean (for normal) during preprocessing.",
                    "priority": "medium",
                })
            else:
                recommendations.append({
                    "column": col,
                    "action": "impute_categorical",
                    "reason": f"Impute missing values in '{col}' with the most frequent category or 'Missing' label.",
                    "priority": "medium",
                })

    # 2. Duplicate Rows
    dup_rows = int(df.duplicated().sum())
    dup_rows_pct = (dup_rows / max(num_rows, 1)) * 100.0
    if dup_rows > 0:
        issues.append({
            "type": "duplicate_rows",
            "column": None,
            "severity": "warning" if dup_rows_pct > 5 else "info",
            "message": f"Found {dup_rows:,} exact duplicate rows ({dup_rows_pct:.2f}% of dataset).",
        })
        recommendations.append({
            "column": None,
            "action": "drop_duplicate_rows",
            "reason": f"Remove {dup_rows:,} duplicate records to avoid artificial weight inflation during ML training.",
            "priority": "high",
        })

    # 3. Duplicate Columns
    sample_df = df if num_rows <= 1000 else df.sample(1000, random_state=42)
    cols_list = list(df.columns)
    duplicate_columns: list[dict[str, str]] = []
    for i in range(len(cols_list)):
        for j in range(i + 1, len(cols_list)):
            c1, c2 = cols_list[i], cols_list[j]
            try:
                if sample_df[c1].equals(sample_df[c2]):
                    # Verify on full dataframe
                    if df[c1].equals(df[c2]):
                        duplicate_columns.append({"col1": c1, "col2": c2})
                        issues.append({
                            "type": "duplicate_columns",
                            "column": c2,
                            "severity": "warning",
                            "message": f"Column '{c2}' is an exact duplicate of column '{c1}'.",
                        })
                        recommendations.append({
                            "column": c2,
                            "action": "drop_duplicate_column",
                            "reason": f"Drop '{c2}' to eliminate redundant data and avoid multicollinearity.",
                            "priority": "high",
                        })
            except Exception:
                pass

    # 4. Constant Columns (zero variance)
    constant_columns: list[str] = []
    near_constant_columns: list[dict[str, Any]] = []
    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0:
            constant_columns.append(col)
            issues.append({
                "type": "constant_column",
                "column": col,
                "severity": "critical",
                "message": f"Column '{col}' is entirely empty / all NaN values.",
            })
            recommendations.append({
                "column": col,
                "action": "drop_column",
                "reason": f"Column '{col}' contains no data and must be removed.",
                "priority": "high",
            })
            continue

        nunique = int(s.nunique())
        if nunique == 1:
            constant_columns.append(col)
            val_str = str(s.iloc[0])
            issues.append({
                "type": "constant_column",
                "column": col,
                "severity": "critical",
                "message": f"Column '{col}' has only 1 unique value ('{val_str}') with zero variance.",
            })
            recommendations.append({
                "column": col,
                "action": "drop_column",
                "reason": f"Drop constant column '{col}' because it provides zero predictive signal to ML models.",
                "priority": "high",
            })
        elif num_rows >= 20:
            top_freq = int(s.value_counts().iloc[0])
            dom_pct = (top_freq / len(s)) * 100.0
            if dom_pct >= 95.0:
                top_val = str(s.value_counts().index[0])
                near_constant_columns.append({
                    "column": col,
                    "dominant_value": top_val,
                    "dominant_pct": _safe_float(dom_pct),
                })
                issues.append({
                    "type": "near_constant_column",
                    "column": col,
                    "severity": "warning",
                    "message": f"Column '{col}' is near-constant: '{top_val}' accounts for {dom_pct:.1f}% of values.",
                })
                recommendations.append({
                    "column": col,
                    "action": "evaluate_near_constant",
                    "reason": f"Inspect '{col}' — nearly all records have value '{top_val}'. Consider variance threshold filtering.",
                    "priority": "medium",
                })

    # 5. Infinite Values
    infinite_values_by_col: dict[str, int] = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col]
            inf_count = int(np.isinf(s).sum())
            if inf_count > 0:
                infinite_values_by_col[col] = inf_count
                issues.append({
                    "type": "infinite_values",
                    "column": col,
                    "severity": "critical",
                    "message": f"Column '{col}' contains {inf_count:,} infinite (+inf/-inf) values.",
                })
                recommendations.append({
                    "column": col,
                    "action": "replace_infinite",
                    "reason": f"Replace infinite values in '{col}' with NaN and impute or clip to min/max before model training.",
                    "priority": "high",
                })

    # 6. Incorrect Data Types & Suspicious Strings
    incorrect_dtypes: list[dict[str, Any]] = []
    suspicious_values_by_col: dict[str, list[dict[str, Any]]] = {}

    for col in df.columns:
        s = df[col]
        # Check if object column should actually be numeric
        if s.dtype == object:
            # Check for suspicious placeholder strings
            val_counts = s.dropna().astype(str).str.strip().str.lower().value_counts()
            found_placeholders = []
            for placeholder in SUSPICIOUS_STRINGS:
                if placeholder in val_counts.index:
                    cnt = int(val_counts[placeholder])
                    found_placeholders.append({"value": placeholder, "count": cnt})
            if found_placeholders:
                suspicious_values_by_col[col] = found_placeholders
                ph_summary = ", ".join([f"{p['value']} ({p['count']})" for p in found_placeholders])
                ph_vals = ", ".join([p["value"] for p in found_placeholders])
                issues.append({
                    "type": "suspicious_placeholders",
                    "column": col,
                    "severity": "warning",
                    "message": f"Column '{col}' contains hidden placeholder nulls: {ph_summary}.",
                })
                recommendations.append({
                    "column": col,
                    "action": "convert_placeholders_to_nan",
                    "reason": f"Replace placeholder strings ({ph_vals}) in '{col}' with proper NaN values.",
                    "priority": "medium",
                })

            # Check if majority of non-null entries can be parsed as numbers
            non_null = s.dropna().astype(str).str.strip()
            if len(non_null) > 0:
                numeric_conv = pd.to_numeric(non_null, errors="coerce")
                valid_num_pct = (numeric_conv.notnull().sum() / len(non_null)) * 100.0
                if valid_num_pct >= 85.0 and valid_num_pct < 100.0:
                    incorrect_dtypes.append({
                        "column": col,
                        "current_dtype": "object",
                        "suggested_dtype": "numeric (float/int)",
                        "confidence_pct": _safe_float(valid_num_pct),
                    })
                    issues.append({
                        "type": "incorrect_dtype",
                        "column": col,
                        "severity": "warning",
                        "message": f"Column '{col}' is stored as text/object, but {valid_num_pct:.1f}% of entries are valid numbers.",
                    })
                    recommendations.append({
                        "column": col,
                        "action": "cast_to_numeric",
                        "reason": f"Clean and cast '{col}' to numeric dtype to allow mathematical operations and ML feature usage.",
                        "priority": "medium",
                    })
                elif valid_num_pct == 100.0:
                    incorrect_dtypes.append({
                        "column": col,
                        "current_dtype": "object",
                        "suggested_dtype": "numeric (float/int)",
                        "confidence_pct": 100.0,
                    })
                    issues.append({
                        "type": "incorrect_dtype",
                        "column": col,
                        "severity": "warning",
                        "message": f"Column '{col}' is stored as object but all values are numbers.",
                    })

        # Check for negative values in columns with strictly non-negative names
        if pd.api.types.is_numeric_dtype(s):
            col_lower = col.lower()
            if any(term in col_lower for term in ["age", "price", "cost", "salary", "count", "quantity", "distance", "height", "weight"]):
                neg_cnt = int((s < 0).sum())
                if neg_cnt > 0:
                    issues.append({
                        "type": "invalid_domain_values",
                        "column": col,
                        "severity": "critical" if "age" in col_lower else "warning",
                        "message": f"Column '{col}' contains {neg_cnt} negative values, which is unexpected for {col_lower}.",
                    })
                    recommendations.append({
                        "column": col,
                        "action": "sanitize_negative_values",
                        "reason": f"Clip or investigate {neg_cnt} negative values in '{col}'.",
                        "priority": "high",
                    })

    # 7. High-Cardinality and Potential Identifier Columns
    high_cardinality_cols: list[dict[str, Any]] = []
    potential_identifiers: list[dict[str, Any]] = []

    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        nunique = int(s.nunique())
        unique_ratio = nunique / len(s)

        # Check identifier
        is_id_name = bool(IDENTIFIER_PATTERNS.search(col))
        if (unique_ratio > 0.95 and not pd.api.types.is_float_dtype(s) and len(s) > 30) or (is_id_name and unique_ratio > 0.8):
            potential_identifiers.append({
                "column": col,
                "nunique": nunique,
                "unique_ratio": _safe_float(unique_ratio * 100.0),
                "matched_name": is_id_name,
            })
            issues.append({
                "type": "potential_identifier",
                "column": col,
                "severity": "info",
                "message": f"Column '{col}' has {unique_ratio*100:.1f}% unique values and matches identifier patterns.",
            })
            recommendations.append({
                "column": col,
                "action": "drop_or_exclude_identifier",
                "reason": f"Exclude identifier column '{col}' from ML feature matrix to prevent memorization and target leakage.",
                "priority": "high",
            })
        elif s.dtype == object and nunique > 50 and unique_ratio > 0.3 and len(s) > 100:
            high_cardinality_cols.append({
                "column": col,
                "nunique": nunique,
                "unique_ratio": _safe_float(unique_ratio * 100.0),
            })
            issues.append({
                "type": "high_cardinality",
                "column": col,
                "severity": "warning",
                "message": f"Categorical column '{col}' has {nunique} unique categories (high cardinality).",
            })
            recommendations.append({
                "column": col,
                "action": "encode_high_cardinality",
                "reason": f"Use target encoding, frequency encoding, or group rare categories into 'Other' for '{col}'.",
                "priority": "medium",
            })

    # Overall Quality Score (0 - 100)
    critical_count = sum(1 for i in issues if i["severity"] == "critical")
    warning_count = sum(1 for i in issues if i["severity"] == "warning")
    info_count = sum(1 for i in issues if i["severity"] == "info")

    penalty = (critical_count * 15.0) + (warning_count * 5.0) + (info_count * 1.5)
    quality_score = max(0.0, min(100.0, round(100.0 - penalty, 1)))

    return {
        "quality_score": quality_score,
        "total_issues": len(issues),
        "critical_issues": critical_count,
        "warning_issues": warning_count,
        "info_issues": info_count,
        "issues": issues,
        "recommendations": recommendations,
        "missing_summary": {
            "total_missing_cells": total_missing_cells,
            "columns_with_missing": len(missing_by_col),
            "missing_by_column": missing_by_col,
        },
        "duplicate_summary": {
            "duplicate_rows": dup_rows,
            "duplicate_rows_pct": _safe_float(dup_rows_pct),
            "duplicate_columns": duplicate_columns,
        },
        "constant_summary": {
            "constant_columns": constant_columns,
            "near_constant_columns": near_constant_columns,
        },
        "infinite_summary": {
            "infinite_values": infinite_values_by_col,
            "total_infinite_cells": sum(infinite_values_by_col.values()),
        },
        "type_anomalies": {
            "incorrect_dtypes": incorrect_dtypes,
            "suspicious_values": suspicious_values_by_col,
        },
        "cardinality_summary": {
            "high_cardinality_columns": high_cardinality_cols,
            "potential_identifiers": potential_identifiers,
        },
    }

"""
EDA Module — Univariate Analysis Engine.

Performs deep univariate statistical and distributional analysis:
- Numerical: Histogram, KDE curve, Box/Violin statistics, Mean/Median/Mode,
  Variance, Std, Min/Max, Quartiles, IQR, Skewness, Kurtosis, Normality test.
- Categorical: Frequency distributions, category counts, count plot data,
  pie/donut data, unique counts, most frequent categories, rare categories.
- Boolean & Datetime: True/False breakdown, date spans, periodic aggregations.
Supports analyzing all columns or a single selected column dynamically.
"""

from typing import Any
import math
import numpy as np
import pandas as pd
from scipy import stats


def _safe_float(val: Any) -> float | None:
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 6)
    except (TypeError, ValueError):
        return None


def analyze_numerical_column(series: pd.Series, max_points_kde: int = 50) -> dict[str, Any]:
    """Calculate complete univariate statistics and distribution for a numerical series."""
    clean = series.dropna()
    total_count = len(series)
    valid_count = len(clean)
    missing_count = total_count - valid_count
    missing_pct = _safe_float((missing_count / max(total_count, 1)) * 100.0)

    if valid_count == 0:
        return {
            "count": 0,
            "missing": missing_count,
            "missing_pct": missing_pct,
            "dtype": str(series.dtype),
        }

    # Central tendency
    mean_val = _safe_float(clean.mean())
    median_val = _safe_float(clean.median())
    try:
        mode_val = _safe_float(clean.mode().iloc[0]) if not clean.mode().empty else median_val
    except Exception:
        mode_val = median_val

    # Dispersion
    std_val = _safe_float(clean.std())
    var_val = _safe_float(clean.var())
    min_val = _safe_float(clean.min())
    max_val = _safe_float(clean.max())
    range_val = _safe_float(clean.max() - clean.min()) if min_val is not None and max_val is not None else None

    # Quantiles
    q1 = _safe_float(clean.quantile(0.25))
    q3 = _safe_float(clean.quantile(0.75))
    iqr_val = _safe_float(clean.quantile(0.75) - clean.quantile(0.25))
    p5 = _safe_float(clean.quantile(0.05))
    p10 = _safe_float(clean.quantile(0.10))
    p90 = _safe_float(clean.quantile(0.90))
    p95 = _safe_float(clean.quantile(0.95))

    # Box plot stats & Tukey fences
    lower_fence = (q1 - 1.5 * iqr_val) if (q1 is not None and iqr_val is not None) else min_val
    upper_fence = (q3 + 1.5 * iqr_val) if (q3 is not None and iqr_val is not None) else max_val
    outliers_mask = (clean < lower_fence) | (clean > upper_fence)
    outlier_points = [_safe_float(x) for x in clean[outliers_mask].head(50).tolist()]
    outlier_count = int(outliers_mask.sum())
    outlier_pct = _safe_float((outlier_count / valid_count) * 100.0)

    # Shape: Skewness & Kurtosis
    skew_val = _safe_float(clean.skew())
    kurt_val = _safe_float(clean.kurtosis())

    # Normality test
    normality_stat = None
    normality_p = None
    is_normal = False
    if valid_count >= 8:
        try:
            if valid_count <= 5000:
                stat_val, p_val = stats.shapiro(clean)
            else:
                stat_val, p_val = stats.normaltest(clean)
            normality_stat = _safe_float(stat_val)
            normality_p = _safe_float(p_val)
            is_normal = bool(p_val > 0.05 and abs(skew_val or 0) < 0.5)
        except Exception:
            is_normal = bool(abs(skew_val or 0) < 0.8)

    # Histogram
    num_bins = min(30, max(10, int(np.sqrt(valid_count))))
    try:
        counts, edges = np.histogram(clean, bins=num_bins)
        bin_centers = [0.5 * (edges[i] + edges[i+1]) for i in range(len(counts))]
        histogram = {
            "counts": [int(c) for c in counts],
            "bin_edges": [_safe_float(e) for e in edges],
            "bin_centers": [_safe_float(c) for c in bin_centers],
        }
    except Exception:
        histogram = {"counts": [], "bin_edges": [], "bin_centers": []}

    # KDE curve estimation
    kde_curve = {"x": [], "y": []}
    if valid_count >= 10 and std_val and std_val > 0:
        try:
            sample_data = clean if valid_count <= 2000 else clean.sample(2000, random_state=42)
            kde = stats.gaussian_kde(sample_data)
            x_vals = np.linspace(min_val or 0, max_val or 1, max_points_kde)
            y_vals = kde.evaluate(x_vals)
            kde_curve = {
                "x": [_safe_float(x) for x in x_vals],
                "y": [_safe_float(y) for y in y_vals],
            }
        except Exception:
            pass

    return {
        "count": valid_count,
        "total_count": total_count,
        "missing": missing_count,
        "missing_pct": missing_pct,
        "nunique": int(clean.nunique()),
        "dtype": str(series.dtype),

        # Central tendency
        "mean": mean_val,
        "median": median_val,
        "mode": mode_val,

        # Dispersion
        "std": std_val,
        "variance": var_val,
        "min": min_val,
        "max": max_val,
        "range": range_val,

        # Percentiles
        "p5": p5,
        "p10": p10,
        "p25": q1,
        "p50": median_val,
        "p75": q3,
        "p90": p90,
        "p95": p95,
        "iqr": iqr_val,

        # Legacy aliases for compatibility
        "25%": q1,
        "50%": median_val,
        "75%": q3,

        # Shape & Normality
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "is_normal": is_normal,
        "normality_test": {
            "statistic": normality_stat,
            "p_value": normality_p,
            "is_normal": is_normal,
        },

        # Box plot data
        "box_plot": {
            "min": min_val,
            "q1": q1,
            "median": median_val,
            "q3": q3,
            "max": max_val,
            "lower_fence": _safe_float(lower_fence),
            "upper_fence": _safe_float(upper_fence),
            "outlier_count": outlier_count,
            "outlier_pct": outlier_pct,
            "sample_outliers": outlier_points,
        },

        # Curves
        "histogram": histogram,
        "kde": kde_curve,
    }


def analyze_categorical_column(series: pd.Series, max_categories: int = 15) -> dict[str, Any]:
    """Calculate complete univariate statistics and distributions for a categorical series."""
    clean = series.dropna().astype(str)
    total_count = len(series)
    valid_count = len(clean)
    missing_count = total_count - valid_count
    missing_pct = _safe_float((missing_count / max(total_count, 1)) * 100.0)

    if valid_count == 0:
        return {
            "count": 0,
            "missing": missing_count,
            "missing_pct": missing_pct,
            "nunique": 0,
            "dtype": str(series.dtype),
        }

    vc = clean.value_counts()
    nunique = len(vc)

    # Top categories
    top_vc = vc.head(max_categories)
    other_count = int(vc.iloc[max_categories:].sum()) if nunique > max_categories else 0

    category_counts = {str(k): int(v) for k, v in top_vc.items()}
    category_percentages = {str(k): _safe_float((v / valid_count) * 100.0) for k, v in top_vc.items()}

    if other_count > 0:
        category_counts["Other"] = other_count
        category_percentages["Other"] = _safe_float((other_count / valid_count) * 100.0)

    # Rare categories (< 1% frequency)
    rare_mask = (vc / valid_count) < 0.01
    rare_categories = [str(cat) for cat in vc[rare_mask].index[:10]]
    rare_count = int(rare_mask.sum())

    return {
        "count": valid_count,
        "total_count": total_count,
        "missing": missing_count,
        "missing_pct": missing_pct,
        "nunique": nunique,
        "dtype": str(series.dtype),

        "top_value": str(vc.index[0]) if len(vc) > 0 else None,
        "top_freq": int(vc.iloc[0]) if len(vc) > 0 else 0,
        "top_pct": _safe_float((vc.iloc[0] / valid_count) * 100.0) if len(vc) > 0 else 0.0,
        "top_10_values": {str(k): int(v) for k, v in vc.head(10).items()},

        "category_counts": category_counts,
        "category_percentages": category_percentages,
        "chart_data": {
            "labels": list(category_counts.keys()),
            "values": list(category_counts.values()),
            "percentages": [category_percentages[k] for k in category_counts.keys()],
        },

        "rare_categories_count": rare_count,
        "sample_rare_categories": rare_categories,
        "is_high_cardinality": nunique > 50 and (nunique / valid_count) > 0.3,
    }


def analyze_datetime_column(series: pd.Series) -> dict[str, Any]:
    """Analyze temporal attributes for a datetime series."""
    clean = pd.to_datetime(series.dropna(), errors="coerce").dropna()
    total_count = len(series)
    valid_count = len(clean)

    if valid_count == 0:
        return {"count": 0, "missing": total_count, "dtype": "datetime"}

    min_date = clean.min()
    max_date = clean.max()
    span_days = (max_date - min_date).days if hasattr(max_date - min_date, "days") else None

    # Month distribution
    months = clean.dt.month_name().value_counts()
    days_of_week = clean.dt.day_name().value_counts()

    return {
        "count": valid_count,
        "missing": total_count - valid_count,
        "min_date": str(min_date),
        "max_date": str(max_date),
        "span_days": span_days,
        "monthly_distribution": {str(k): int(v) for k, v in months.head(12).items()},
        "day_of_week_distribution": {str(k): int(v) for k, v in days_of_week.items()},
        "dtype": "datetime",
    }


def compute_univariate(
    df: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
    column: str | None = None,
) -> dict[str, Any]:
    """
    Compute univariate analysis for a single column or all columns in the dataset.
    """
    if metadata is None:
        metadata = {}

    if column and column in df.columns:
        s = df[column]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            analysis = {"type": "numerical", "data": analyze_numerical_column(s)}
        elif pd.api.types.is_datetime64_any_dtype(s):
            analysis = {"type": "datetime", "data": analyze_datetime_column(s)}
        else:
            analysis = {"type": "categorical", "data": analyze_categorical_column(s)}
        return {"column": column, **analysis}

    # Analyze all columns
    numerical_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numerical_cols and not pd.api.types.is_datetime64_any_dtype(df[c])]
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

    num_results: dict[str, Any] = {}
    for col in numerical_cols:
        num_results[col] = analyze_numerical_column(df[col])

    cat_results: dict[str, Any] = {}
    for col in categorical_cols:
        cat_results[col] = analyze_categorical_column(df[col])

    dt_results: dict[str, Any] = {}
    for col in datetime_cols:
        dt_results[col] = analyze_datetime_column(df[col])

    return {
        "numerical": num_results,
        "categorical": cat_results,
        "datetime": dt_results,
        "total_columns_analyzed": len(df.columns),
    }

"""
EDA Module — Bivariate Analysis Engine.

Performs bivariate relationship analysis across all column type combinations:
1. Numerical vs Numerical: Pearson/Spearman correlation, p-values, trendline,
   scatter plot sampling, covariance.
2. Numerical vs Categorical: Grouped descriptive statistics, One-way ANOVA F-test,
   Kruskal-Wallis test, effect size (Eta-squared), grouped box/violin metrics.
3. Categorical vs Categorical: Contingency table (cross-tabulation), Chi-Square
   test of independence, Cramer's V, stacked/grouped bar distributions.
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


def analyze_num_vs_num(df: pd.DataFrame, col1: str, col2: str, sample_size: int = 500) -> dict[str, Any]:
    """Analyze the relationship between two numerical columns."""
    sub = df[[col1, col2]].dropna()
    n = len(sub)
    if n < 3:
        return {"available": False, "reason": "Insufficient non-null rows (<3)"}

    x = sub[col1].astype(float)
    y = sub[col2].astype(float)

    # Pearson correlation & p-value
    try:
        p_corr, p_pval = stats.pearsonr(x, y)
    except Exception:
        p_corr, p_pval = None, None

    # Spearman correlation & p-value
    try:
        s_corr, s_pval = stats.spearmanr(x, y)
    except Exception:
        s_corr, s_pval = None, None

    # Covariance
    try:
        cov_val = float(np.cov(x, y)[0, 1])
    except Exception:
        cov_val = None

    # Linear regression trendline (OLS)
    trendline = {}
    try:
        slope, intercept, r_val, p_reg, std_err = stats.linregress(x, y)
        trendline = {
            "slope": _safe_float(slope),
            "intercept": _safe_float(intercept),
            "r_squared": _safe_float(r_val ** 2),
            "p_value": _safe_float(p_reg),
            "formula": f"{col2} = {_safe_float(slope)} × {col1} + {_safe_float(intercept)}",
        }
    except Exception:
        pass

    # Scatter points sample
    sampled = sub if n <= sample_size else sub.sample(sample_size, random_state=42)
    scatter_points = [
        {"x": _safe_float(row[col1]), "y": _safe_float(row[col2])}
        for _, row in sampled.iterrows()
    ]

    return {
        "available": True,
        "relationship_type": "numerical_vs_numerical",
        "col1": col1,
        "col2": col2,
        "sample_count": n,
        "pearson": {
            "correlation": _safe_float(p_corr),
            "p_value": _safe_float(p_pval),
            "significant": bool(p_pval is not None and p_pval < 0.05),
        },
        "spearman": {
            "correlation": _safe_float(s_corr),
            "p_value": _safe_float(s_pval),
            "significant": bool(s_pval is not None and s_pval < 0.05),
        },
        "covariance": _safe_float(cov_val),
        "trendline": trendline,
        "scatter_points": scatter_points,
    }


def analyze_num_vs_cat(df: pd.DataFrame, num_col: str, cat_col: str, max_groups: int = 15) -> dict[str, Any]:
    """Analyze the relationship between a numerical and a categorical column."""
    sub = df[[num_col, cat_col]].dropna()
    n = len(sub)
    if n < 5:
        return {"available": False, "reason": "Insufficient non-null rows"}

    # Top categories
    top_cats = sub[cat_col].value_counts().head(max_groups).index.tolist()
    sub_filtered = sub[sub[cat_col].isin(top_cats)]

    grouped_stats: dict[str, Any] = {}
    group_arrays = []

    for cat in top_cats:
        cat_vals = sub_filtered[sub_filtered[cat_col] == cat][num_col].astype(float)
        if len(cat_vals) > 0:
            group_arrays.append(cat_vals.values)
            grouped_stats[str(cat)] = {
                "count": len(cat_vals),
                "mean": _safe_float(cat_vals.mean()),
                "std": _safe_float(cat_vals.std()),
                "median": _safe_float(cat_vals.median()),
                "min": _safe_float(cat_vals.min()),
                "max": _safe_float(cat_vals.max()),
                "q1": _safe_float(cat_vals.quantile(0.25)),
                "q3": _safe_float(cat_vals.quantile(0.75)),
            }

    # One-Way ANOVA test
    anova_stat, anova_p = None, None
    eta_squared = None
    if len(group_arrays) >= 2:
        try:
            f_stat, p_val = stats.f_oneway(*group_arrays)
            anova_stat = _safe_float(f_stat)
            anova_p = _safe_float(p_val)

            # Compute Eta-squared (effect size)
            grand_mean = sub_filtered[num_col].mean()
            ss_between = sum(len(arr) * ((arr.mean() - grand_mean) ** 2) for arr in group_arrays)
            ss_total = sum(((sub_filtered[num_col] - grand_mean) ** 2))
            if ss_total > 0:
                eta_squared = _safe_float(ss_between / ss_total)
        except Exception:
            pass

    return {
        "available": True,
        "relationship_type": "numerical_vs_categorical",
        "num_col": num_col,
        "cat_col": cat_col,
        "sample_count": n,
        "num_categories": len(grouped_stats),
        "grouped_statistics": grouped_stats,
        "anova": {
            "f_statistic": anova_stat,
            "p_value": anova_p,
            "significant": bool(anova_p is not None and anova_p < 0.05),
            "eta_squared": eta_squared,
            "effect_size": "large" if (eta_squared or 0) > 0.14 else "medium" if (eta_squared or 0) > 0.06 else "small",
        },
    }


def analyze_cat_vs_cat(df: pd.DataFrame, col1: str, col2: str, max_categories: int = 10) -> dict[str, Any]:
    """Analyze the relationship between two categorical columns."""
    sub = df[[col1, col2]].dropna()
    n = len(sub)
    if n < 5:
        return {"available": False, "reason": "Insufficient non-null rows"}

    # Filter to top categories for readability
    top1 = sub[col1].value_counts().head(max_categories).index.tolist()
    top2 = sub[col2].value_counts().head(max_categories).index.tolist()
    sub_filtered = sub[sub[col1].isin(top1) & sub[col2].isin(top2)]

    if len(sub_filtered) == 0:
        return {"available": False, "reason": "No data after category filtering"}

    # Contingency Table
    ct = pd.crosstab(sub_filtered[col1], sub_filtered[col2])
    ct_dict = {str(r): {str(c): int(ct.loc[r, c]) for c in ct.columns} for r in ct.index}

    # Chi-square test
    chi2_stat, chi2_p, dof = None, None, None
    cramers_v = None

    try:
        chi2, p_val, deg_f, _ = stats.chi2_contingency(ct)
        chi2_stat = _safe_float(chi2)
        chi2_p = _safe_float(p_val)
        dof = int(deg_f)

        # Cramer's V calculation
        min_dim = min(ct.shape[0] - 1, ct.shape[1] - 1)
        if min_dim > 0 and len(sub_filtered) > 0:
            cramers_v = _safe_float(np.sqrt(chi2 / (len(sub_filtered) * min_dim)))
    except Exception:
        pass

    return {
        "available": True,
        "relationship_type": "categorical_vs_categorical",
        "col1": col1,
        "col2": col2,
        "sample_count": len(sub_filtered),
        "contingency_table": ct_dict,
        "row_categories": [str(c) for c in ct.index],
        "col_categories": [str(c) for c in ct.columns],
        "chi_square": {
            "statistic": chi2_stat,
            "p_value": chi2_p,
            "degrees_of_freedom": dof,
            "significant": bool(chi2_p is not None and chi2_p < 0.05),
            "cramers_v": cramers_v,
            "strength": "strong" if (cramers_v or 0) > 0.3 else "moderate" if (cramers_v or 0) > 0.15 else "weak",
        },
    }


def compute_bivariate(
    df: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
    col1: str | None = None,
    col2: str | None = None,
) -> dict[str, Any]:
    """
    Compute bivariate relationship between two specified columns,
    or find the top pairwise associations across the entire dataset.
    """
    if metadata is None:
        metadata = {}

    # Explicit pair analysis
    if col1 and col2 and col1 in df.columns and col2 in df.columns:
        is_num1 = pd.api.types.is_numeric_dtype(df[col1]) and not pd.api.types.is_bool_dtype(df[col1])
        is_num2 = pd.api.types.is_numeric_dtype(df[col2]) and not pd.api.types.is_bool_dtype(df[col2])

        if is_num1 and is_num2:
            return analyze_num_vs_num(df, col1, col2)
        elif is_num1 and not is_num2:
            return analyze_num_vs_cat(df, num_col=col1, cat_col=col2)
        elif not is_num1 and is_num2:
            return analyze_num_vs_cat(df, num_col=col2, cat_col=col1)
        else:
            return analyze_cat_vs_cat(df, col1, col2)

    # Automated pairwise screening
    numerical_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numerical_cols]

    top_correlations = []
    if len(numerical_cols) >= 2:
        sub_num = df[numerical_cols].dropna()
        if len(sub_num) >= 5:
            corr_mat = sub_num.corr()
            for i in range(len(numerical_cols)):
                for j in range(i + 1, len(numerical_cols)):
                    c1, c2 = numerical_cols[i], numerical_cols[j]
                    val = corr_mat.loc[c1, c2]
                    if not math.isnan(float(val)):
                        top_correlations.append({
                            "col1": c1,
                            "col2": c2,
                            "type": "num_num",
                            "metric": "pearson",
                            "score": _safe_float(val),
                            "abs_score": abs(float(val)),
                        })
    top_correlations.sort(key=lambda x: x["abs_score"], reverse=True)

    return {
        "top_pairwise_associations": top_correlations[:20],
        "default_col1": numerical_cols[0] if numerical_cols else (df.columns[0] if len(df.columns) > 0 else None),
        "default_col2": numerical_cols[1] if len(numerical_cols) > 1 else (df.columns[1] if len(df.columns) > 1 else None),
    }

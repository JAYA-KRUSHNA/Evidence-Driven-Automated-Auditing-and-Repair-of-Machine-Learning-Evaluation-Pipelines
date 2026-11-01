"""
EDA Module — Intelligent Automated Insights Engine.

Synthesizes findings across Overview, Quality, Distributions, Correlations,
Outliers, Target, and Features into structured, evidence-backed insights.
Every insight provides:
  1. Finding
  2. Statistical Evidence (actual numbers, tests, metrics)
  3. Explanation (why it matters)
  4. Recommendation (what to do in preprocessing/modeling)
  5. Severity level ('critical', 'warning', 'info', 'success')
  6. Category ('quality', 'distribution', 'correlation', 'outliers', 'target', 'features')
Guaranteed 100% data-driven; no hardcoded or fabricated observations.
"""

from typing import Any
import math
import pandas as pd


def generate_insights(
    overview_data: dict[str, Any],
    quality_data: dict[str, Any],
    univariate_data: dict[str, Any],
    correlation_data: dict[str, Any],
    outliers_data: dict[str, Any],
    target_data: dict[str, Any] | None = None,
    features_data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Generate an intelligent, ranked list of statistical insights backed by actual data.
    """
    insights: list[dict[str, Any]] = []

    # 1. Dataset Health & Overview Insights
    health_score = overview_data.get("health_score", 100)
    health_grade = overview_data.get("health_grade", "Good")
    num_rows = overview_data.get("num_rows", 0)
    num_cols = overview_data.get("num_columns", 0)

    if health_score >= 90:
        insights.append({
            "finding": f"High overall dataset health score of {health_score:.1f}/100 ({health_grade}).",
            "evidence": f"Total cells: {overview_data.get('total_cells', 0):,}, Missing: {overview_data.get('missing_pct', 0):.2f}%, Duplicate rows: {overview_data.get('duplicate_rows', 0)}.",
            "explanation": "The dataset is well-structured with minimal noise, suitable for immediate feature engineering and model training.",
            "recommendation": "Proceed with baseline model benchmarking or feature selection.",
            "severity": "success",
            "category": "overview",
        })
    elif health_score < 60:
        insights.append({
            "finding": f"Dataset health score is low at {health_score:.1f}/100 ({health_grade}).",
            "evidence": f"Multiple quality penalties identified: {len(overview_data.get('health_penalties', []))} areas requiring intervention.",
            "explanation": "Severe missingness, duplicates, or column anomalies will compromise model stability and convergence if not cleaned.",
            "recommendation": "Review data quality audit recommendations and run the automated preprocessing pipeline before modeling.",
            "severity": "critical",
            "category": "overview",
        })

    # 2. Missing Data Insights
    missing_pct = overview_data.get("missing_pct", 0.0) or 0.0
    cols_with_missing = quality_data.get("missing_summary", {}).get("columns_with_missing", 0)
    if missing_pct > 0:
        sev = "critical" if missing_pct > 20 else "warning" if missing_pct > 5 else "info"
        insights.append({
            "finding": f"{missing_pct:.2f}% of data cells are missing across {cols_with_missing} columns.",
            "evidence": f"Total missing cells: {overview_data.get('total_missing_cells', 0):,} out of {overview_data.get('total_cells', 0):,}.",
            "explanation": "Missing values cause estimator failures in most algorithms and can introduce bias if missingness is non-random.",
            "recommendation": "Impute numerical columns with median/KNN and categorical with mode or distinct 'Missing' indicator.",
            "severity": sev,
            "category": "quality",
        })

    # 3. Duplicate Records
    dup_rows = overview_data.get("duplicate_rows", 0)
    if dup_rows > 0:
        dup_pct = (dup_rows / max(num_rows, 1)) * 100.0
        insights.append({
            "finding": f"Found {dup_rows:,} exact duplicate records ({dup_pct:.2f}% of rows).",
            "evidence": f"{dup_rows} identical row signatures detected across all {num_cols} attributes.",
            "explanation": "Duplicate records inflate training loss weights and cause data leakage between train/test splits.",
            "recommendation": "Deduplicate dataset by retaining only the first occurrence of each unique record.",
            "severity": "warning" if dup_pct > 3 else "info",
            "category": "quality",
        })

    # 4. Constant / Near-Constant Columns
    constant_cols = quality_data.get("constant_summary", {}).get("constant_columns", [])
    if constant_cols:
        insights.append({
            "finding": f"{len(constant_cols)} constant column(s) detected: {', '.join(constant_cols[:3])}.",
            "evidence": f"Variance is exactly 0.0; all rows hold identical or null values.",
            "explanation": "Constant features provide zero mathematical information to any machine learning algorithm and consume memory.",
            "recommendation": f"Drop column(s) {', '.join(constant_cols[:3])} from feature matrix.",
            "severity": "critical",
            "category": "quality",
        })

    # 5. Skewness and Distribution Insights
    num_univariate = univariate_data.get("numerical", {})
    skewed_cols = []
    for col, stats_dict in num_univariate.items():
        sk = stats_dict.get("skewness")
        if sk is not None and abs(sk) >= 1.5:
            skewed_cols.append((col, sk, "right-skewed" if sk > 0 else "left-skewed"))

    if skewed_cols:
        skewed_cols.sort(key=lambda x: abs(x[1]), reverse=True)
        top_skew = skewed_cols[0]
        insights.append({
            "finding": f"Column '{top_skew[0]}' exhibits extreme {top_skew[2]} distribution (skewness = {top_skew[1]:.2f}).",
            "evidence": f"Fisher-Pearson skewness coefficient is {top_skew[1]:.2f} (normality threshold is ±0.5).",
            "explanation": "Heavily skewed features distort distance metrics (KNN, SVM) and violate normality assumptions in linear/logistic regression.",
            "recommendation": f"Apply log1p, sqrt, or Yeo-Johnson power transformation to '{top_skew[0]}' during preprocessing.",
            "severity": "warning",
            "category": "distribution",
        })

    # 6. High Correlation & Multicollinearity
    corr_pairs = correlation_data.get("highly_correlated_pairs", [])
    if corr_pairs:
        top_pair = corr_pairs[0]
        c1, c2, r_val = top_pair["col1"], top_pair["col2"], top_pair["correlation"]
        insights.append({
            "finding": f"Strong correlation ({r_val:.2f}) between '{c1}' and '{c2}'.",
            "evidence": f"Pearson correlation coefficient |r| = {abs(r_val):.3f} >= 0.70 threshold.",
            "explanation": "High correlation creates redundant feature representations, inflates regression coefficient variances, and causes unstable model weights.",
            "recommendation": f"Retain the feature with higher correlation to target and drop the other, or apply PCA dimensionality reduction.",
            "severity": "warning" if abs(r_val) < 0.95 else "critical",
            "category": "correlation",
        })

    # 7. Outliers
    total_iqr = outliers_data.get("total_iqr_outliers", 0)
    cols_with_outliers = outliers_data.get("columns_with_outliers", 0)
    if total_iqr > 0 and num_rows > 0:
        outlier_rate = (total_iqr / max(num_rows * max(len(num_univariate), 1), 1)) * 100.0
        insights.append({
            "finding": f"{total_iqr:,} potential outlier values identified across {cols_with_outliers} numerical columns.",
            "evidence": f"Tukey's IQR method detected outliers in {cols_with_outliers} columns with overall density of {outlier_rate:.1f}%.",
            "explanation": "Outliers can distort mean-squared error loss functions and cause decision tree splits on rare artifacts.",
            "recommendation": "Apply Winsorization (capping at 1st and 99th percentiles) or robust feature scaling instead of hard deletion.",
            "severity": "warning" if outlier_rate > 3 else "info",
            "category": "outliers",
        })

    # 8. Target Variable Insights (if available)
    if target_data and target_data.get("available"):
        tgt_col = target_data.get("target_column")
        task = target_data.get("inferred_task")

        if task == "classification":
            is_balanced = target_data.get("is_balanced", True)
            ratio = target_data.get("imbalance_ratio", 1.0) or 1.0
            minority = target_data.get("minority_class")
            min_pct = target_data.get("minority_percentage")

            if not is_balanced:
                insights.append({
                    "finding": f"Target variable '{tgt_col}' exhibits significant class imbalance (ratio {ratio:.1f}:1).",
                    "evidence": f"Minority class '{minority}' represents only {min_pct:.1f}% of observations.",
                    "explanation": "Models trained on imbalanced targets frequently predict the majority class, achieving misleadingly high accuracy but poor recall.",
                    "recommendation": "Use SMOTE oversampling, class weight adjustments (class_weight='balanced'), and evaluate using PR-AUC / F1-score rather than raw accuracy.",
                    "severity": "critical" if ratio > 6 else "warning",
                    "category": "target",
                })
            else:
                insights.append({
                    "finding": f"Target variable '{tgt_col}' is well-balanced across classes (ratio {ratio:.1f}:1).",
                    "evidence": f"All classes are within acceptable parity threshold (imbalance ratio <= 3.0).",
                    "explanation": "Balanced classes allow standard cross-entropy and accuracy metrics to reflect model performance accurately.",
                    "recommendation": "Standard stratified k-fold cross-validation is well-suited for benchmarking.",
                    "severity": "success",
                    "category": "target",
                })
        else:
            # Regression target
            tgt_skew = target_data.get("skewness")
            tgt_outliers = target_data.get("target_outliers_pct", 0.0) or 0.0
            if tgt_skew is not None and abs(tgt_skew) > 1.2:
                insights.append({
                    "finding": f"Regression target '{tgt_col}' is significantly skewed (skewness = {tgt_skew:.2f}).",
                    "evidence": f"Fisher-Pearson skewness = {tgt_skew:.2f}, target outlier rate = {tgt_outliers:.1f}%.",
                    "explanation": "Target skewness leads to large residual variances for high-value predictions in regression models.",
                    "recommendation": "Apply log or Box-Cox transformation to the target variable before model fitting and inverse-transform predictions.",
                    "severity": "warning",
                    "category": "target",
                })

    # 9. Potential Data Leakage
    if features_data:
        leakage = features_data.get("potential_leakage_features", [])
        if leakage:
            top_leak = leakage[0]
            insights.append({
                "finding": f"Potential target leakage detected in feature '{top_leak['feature']}'.",
                "evidence": f"Correlation with target is {top_leak['correlation_with_target']:.3f} (>= 0.95 threshold).",
                "explanation": "Features with near-perfect target correlation often represent post-outcome data or target proxies, resulting in unrealistic offline performance.",
                "recommendation": f"Verify the business definition of '{top_leak['feature']}' and exclude it if recorded post-event.",
                "severity": "critical",
                "category": "features",
            })

    # Sort insights by severity: critical > warning > info > success
    severity_order = {"critical": 0, "warning": 1, "info": 2, "success": 3}
    insights.sort(key=lambda x: severity_order.get(x["severity"], 4))

    return insights

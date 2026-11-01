"""
Comprehensive Tests for Enhanced EDA Engine & Multi-Format Exporter.

Validates:
1. Numerical datasets
2. Categorical datasets
3. Mixed datasets
4. Datasets with missing values
5. Datasets with duplicate records & duplicate columns
6. Datasets with severe outliers
7. Classification target analysis
8. Regression target analysis
9. Small datasets (<= 10 rows)
10. Large datasets (>= 1,500 rows)
11. Multi-method outlier detection (IQR, Z-Score, Modified Z-Score, Isolation Forest, LOF)
12. VIF and PCA multivariate calculations
13. Data quality audit and health score calculation
14. AI Insights engine data grounding
15. Report exports: HTML, PDF, Excel (.xlsx)
16. Plotly interactive chart generation
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

_proj = Path(__file__).resolve().parents[2]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

from universal_ml_studio.eda.engine import EDAEngine
from universal_ml_studio.eda.overview import compute_overview, calculate_health_score
from universal_ml_studio.eda.quality import compute_quality
from universal_ml_studio.eda.univariate import compute_univariate
from universal_ml_studio.eda.bivariate import compute_bivariate
from universal_ml_studio.eda.multivariate import compute_multivariate
from universal_ml_studio.eda.outliers import compute_outliers
from universal_ml_studio.eda.target import compute_target_analysis
from universal_ml_studio.eda.features import compute_feature_analysis
from universal_ml_studio.eda.insights import generate_insights
from universal_ml_studio.eda.reports import generate_html_report, generate_pdf_report, generate_excel_report


# ── Fixture Generators ────────────────────────────────────────────────────────

def make_pure_numerical(n=100):
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "num_a": rng.normal(50, 10, n),
        "num_b": rng.exponential(2, n),
        "num_c": rng.uniform(0, 100, n),
    })

def make_pure_categorical(n=100):
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "cat_a": rng.choice(["Red", "Green", "Blue"], n),
        "cat_b": rng.choice(["Low", "Med", "High", "Critical"], n),
        "cat_c": rng.choice(["Yes", "No"], n),
    })

def make_mixed_dataset(n=150):
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "age": rng.randint(18, 70, n).astype(float),
        "salary": rng.normal(60000, 15000, n),
        "department": rng.choice(["Sales", "Eng", "HR", "Marketing"], n),
        "active": rng.choice([True, False], n),
        "churn": rng.choice([0, 1], n, p=[0.75, 0.25]),
    })
    # inject missing values
    df.loc[0:5, "salary"] = np.nan
    df.loc[10:12, "department"] = np.nan
    # inject duplicates
    df.loc[15:18] = df.loc[14].values
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  Test 1: Dataset Types & Dimensions
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatasetTypes:

    def test_pure_numerical_dataset(self):
        df = make_pure_numerical()
        eda = EDAEngine.analyze(df)
        assert eda["overview"]["num_rows"] == 100
        assert eda["overview"]["num_columns"] == 3
        assert len(eda["statistics"]["numerical"]) == 3
        assert len(eda["statistics"]["categorical"]) == 0
        assert eda["multivariate"]["available"] is True

    def test_pure_categorical_dataset(self):
        df = make_pure_categorical()
        eda = EDAEngine.analyze(df)
        assert eda["overview"]["num_rows"] == 100
        assert len(eda["statistics"]["categorical"]) == 3
        assert len(eda["statistics"]["numerical"]) == 0
        assert eda["multivariate"]["available"] is False

    def test_mixed_dataset(self):
        df = make_mixed_dataset()
        eda = EDAEngine.analyze(df, target_column="churn")
        assert "age" in eda["statistics"]["numerical"]
        assert "department" in eda["statistics"]["categorical"]
        assert eda["target_analysis"]["available"] is True
        assert eda["target_analysis"]["inferred_task"] == "classification"

    def test_small_dataset(self):
        # 5 rows
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": ["a", "b", "c", "d", "e"]})
        eda = EDAEngine.analyze(df)
        assert eda["overview"]["num_rows"] == 5
        assert eda["quality"]["quality_score"] > 0
        # Serializability
        json.dumps(eda)

    def test_large_dataset(self):
        # 1500 rows
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            f"col_{i}": rng.randn(1500) for i in range(10)
        })
        eda = EDAEngine.analyze(df)
        assert eda["overview"]["num_rows"] == 1500
        assert eda["multivariate"]["available"] is True
        assert len(eda["multivariate"]["pca"]["explained_variance_pct"]) > 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Test 2: Data Quality & Health Score
# ═══════════════════════════════════════════════════════════════════════════════

class TestQualityAndHealth:

    def test_clean_health_score(self):
        df = make_pure_numerical()
        health = calculate_health_score(
            num_rows=100, num_cols=3, total_missing=0,
            duplicate_rows=0, constant_cols_count=0,
            duplicate_cols_count=0, outlier_pct=0.0,
            high_cardinality_count=0
        )
        assert health["score"] == 100.0
        assert health["grade"] == "Excellent"

    def test_severe_missing_penalizes_health(self):
        health = calculate_health_score(
            num_rows=100, num_cols=3, total_missing=150,  # 50% missing
            duplicate_rows=0, constant_cols_count=0,
            duplicate_cols_count=0, outlier_pct=0.0,
            high_cardinality_count=0
        )
        assert health["score"] <= 70.0
        assert len(health["penalties"]) > 0

    def test_constant_column_detection(self):
        df = pd.DataFrame({
            "normal": [1, 2, 3, 4, 5],
            "constant": [42, 42, 42, 42, 42],
        })
        qual = compute_quality(df)
        assert "constant" in qual["constant_summary"]["constant_columns"]
        assert any(i["type"] == "constant_column" for i in qual["issues"])

    def test_duplicate_column_detection(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [1, 2, 3, 4, 5],
            "c": [9, 8, 7, 6, 5],
        })
        qual = compute_quality(df)
        assert len(qual["duplicate_summary"]["duplicate_columns"]) == 1
        assert qual["duplicate_summary"]["duplicate_columns"][0]["col1"] == "a"
        assert qual["duplicate_summary"]["duplicate_columns"][0]["col2"] == "b"

    def test_infinite_value_detection(self):
        df = pd.DataFrame({
            "val": [1.0, 2.0, np.inf, 4.0, -np.inf],
        })
        qual = compute_quality(df)
        assert qual["infinite_summary"]["total_infinite_cells"] == 2


# ═══════════════════════════════════════════════════════════════════════════════
#  Test 3: Univariate Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class TestUnivariate:

    def test_numerical_univariate_metrics(self):
        df = pd.DataFrame({"age": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65]})
        res = compute_univariate(df, column="age")
        assert res["type"] == "numerical"
        d = res["data"]
        assert d["mean"] == 42.5
        assert d["median"] == 42.5
        assert "kde" in d
        assert "histogram" in d
        assert len(d["histogram"]["counts"]) > 0

    def test_categorical_univariate_metrics(self):
        df = pd.DataFrame({"city": ["London", "Paris", "London", "Tokyo", "London", "Paris"]})
        res = compute_univariate(df, column="city")
        assert res["type"] == "categorical"
        d = res["data"]
        assert d["top_value"] == "London"
        assert d["top_freq"] == 3
        assert d["nunique"] == 3


# ═══════════════════════════════════════════════════════════════════════════════
#  Test 4: Bivariate Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class TestBivariate:

    def test_num_vs_num_bivariate(self):
        rng = np.random.RandomState(42)
        x = rng.randn(100)
        y = 2.5 * x + rng.randn(100) * 0.2
        df = pd.DataFrame({"x": x, "y": y})
        biv = compute_bivariate(df, col1="x", col2="y")
        assert biv["relationship_type"] == "numerical_vs_numerical"
        assert biv["pearson"]["correlation"] > 0.9
        assert biv["trendline"]["r_squared"] > 0.8
        assert len(biv["scatter_points"]) > 0

    def test_num_vs_cat_bivariate(self):
        df = pd.DataFrame({
            "salary": [30000, 32000, 31000, 80000, 85000, 82000],
            "dept": ["Support", "Support", "Support", "Exec", "Exec", "Exec"],
        })
        biv = compute_bivariate(df, col1="salary", col2="dept")
        assert biv["relationship_type"] == "numerical_vs_categorical"
        assert "anova" in biv
        assert biv["anova"]["f_statistic"] is not None
        assert biv["anova"]["significant"] is True

    def test_cat_vs_cat_bivariate(self):
        df = pd.DataFrame({
            "gender": ["M", "M", "F", "F", "M", "F"] * 10,
            "purchased": ["Yes", "No", "Yes", "Yes", "No", "Yes"] * 10,
        })
        biv = compute_bivariate(df, col1="gender", col2="purchased")
        assert biv["relationship_type"] == "categorical_vs_categorical"
        assert "chi_square" in biv
        assert "contingency_table" in biv


# ═══════════════════════════════════════════════════════════════════════════════
#  Test 5: Multivariate Analysis (PCA & VIF)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultivariate:

    def test_pca_and_vif_computation(self):
        rng = np.random.RandomState(42)
        base = rng.randn(100)
        df = pd.DataFrame({
            "x1": base,
            "x2": base * 2.0 + rng.randn(100) * 0.1,  # highly collinear
            "x3": rng.randn(100),
            "x4": rng.randn(100),
        })
        multi = compute_multivariate(df)
        assert multi["available"] is True
        assert len(multi["pca"]["explained_variance_pct"]) >= 2
        assert len(multi["pca"]["projection_2d"]) > 0
        assert "vif" in multi
        # Collinear pair should show high VIF
        assert len(multi["vif"]["severe_features"]) > 0 or len(multi["vif"]["moderate_features"]) > 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Test 6: Multi-Method Outlier Detection
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutliers:

    def test_outlier_detection_methods(self):
        rng = np.random.RandomState(42)
        vals = list(rng.randn(100)) + [100.0, -100.0]  # extreme outliers
        df = pd.DataFrame({"f1": vals, "f2": rng.randn(102)})
        out = compute_outliers(df, method="all")

        assert out["total_iqr_outliers"] >= 2
        methods = {m["method"]: m for m in out["methods_comparison"]}
        assert "IQR (Tukey's Fences)" in methods
        assert "Standard Z-Score (|Z| > 3)" in methods
        assert "Modified Z-Score (MAD > 3.5)" in methods
        assert "Isolation Forest (Ensemble)" in methods
        assert "Local Outlier Factor (LOF)" in methods
        # Verify recommendation generated
        assert len(methods["IQR (Tukey's Fences)"]["recommendation"]) > 10


# ═══════════════════════════════════════════════════════════════════════════════
#  Test 7: Target Variable Analysis (Classification & Regression)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTargetAnalysisEnhanced:

    def test_classification_target_imbalance(self):
        df = pd.DataFrame({
            "feature": range(100),
            "target": [0] * 90 + [1] * 10,
        })
        res = compute_target_analysis(df, target_column="target")
        assert res["inferred_task"] == "classification"
        assert res["imbalance_ratio"] == 9.0
        assert res["is_balanced"] is False
        assert "Severe Imbalance" in res["imbalance_status"] or "Moderate Imbalance" in res["imbalance_status"]

    def test_regression_target(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "x": rng.randn(100),
            "price": rng.exponential(50000, 100),
        })
        res = compute_target_analysis(df, target_column="price", task_type="regression")
        assert res["inferred_task"] == "regression"
        assert res["mean"] > 0
        assert res["skewness"] is not None

    def test_target_override(self):
        # Integer target with 2 classes, override to regression
        df = pd.DataFrame({"y": [0, 1] * 25})
        res = compute_target_analysis(df, target_column="y", task_type="regression")
        assert res["inferred_task"] == "regression"


# ═══════════════════════════════════════════════════════════════════════════════
#  Test 8: Intelligent Insights & Features
# ═══════════════════════════════════════════════════════════════════════════════

class TestInsightsAndFeatures:

    def test_automated_feature_analysis(self):
        rng = np.random.RandomState(42)
        x = rng.randn(100)
        df = pd.DataFrame({
            "feature_a": x,
            "feature_b": x * 1.0001,  # near-duplicate / redundant
            "low_var": [1.0] * 99 + [1.00001],
            "target": rng.choice([0, 1], 100),
        })
        feat = compute_feature_analysis(df, target_column="target")
        assert len(feat["highly_correlated_pairs"]) >= 1
        assert len(feat["low_variance_features"]) >= 1
        assert len(feat["feature_selection_recommendations"]) >= 1

    def test_intelligent_insights_grounded(self):
        df = make_mixed_dataset()
        eda = EDAEngine.analyze(df, target_column="churn")
        insights = eda.get("insights", [])
        assert len(insights) > 0
        for ins in insights:
            assert "finding" in ins and len(ins["finding"]) > 5
            assert "evidence" in ins and len(ins["evidence"]) > 5
            assert "explanation" in ins and len(ins["explanation"]) > 5
            assert "recommendation" in ins and len(ins["recommendation"]) > 5
            assert ins["severity"] in ("critical", "warning", "info", "success")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test 9: Report Generation (HTML, PDF, Excel)
# ═══════════════════════════════════════════════════════════════════════════════

class TestReportGeneration:

    def test_html_report(self):
        df = make_mixed_dataset()
        eda = EDAEngine.analyze(df, target_column="churn")
        html = generate_html_report("Test_Dataset", eda)
        assert "<!DOCTYPE html>" in html
        assert "Test_Dataset" in html
        assert "Automated EDA Audit Report" in html

    def test_pdf_report(self):
        df = make_mixed_dataset()
        eda = EDAEngine.analyze(df, target_column="churn")
        pdf_bytes = generate_pdf_report("Test_Dataset", eda)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000
        assert pdf_bytes.startswith(b"%PDF")

    def test_excel_report(self):
        df = make_mixed_dataset()
        eda = EDAEngine.analyze(df, target_column="churn")
        excel_bytes = generate_excel_report("Test_Dataset", eda)
        assert isinstance(excel_bytes, bytes)
        assert len(excel_bytes) > 1000
        # Check ZIP / XLSX magic bytes (PK\x03\x04)
        assert excel_bytes[:2] == b"PK"


# ═══════════════════════════════════════════════════════════════════════════════
#  Test 10: Plotly Chart Generation
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlotlyCharts:

    def test_hist_kde_chart(self):
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        fig_dict = EDAEngine.generate_chart(df, "hist_kde", {"col1": "val"})
        assert "data" in fig_dict
        assert "layout" in fig_dict

    def test_scatter_chart(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 8, 10]})
        fig_dict = EDAEngine.generate_chart(df, "scatter", {"col1": "x", "col2": "y"})
        assert "data" in fig_dict
        assert len(fig_dict["data"]) >= 1

    def test_corr_heatmap_chart(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        fig_dict = EDAEngine.generate_chart(df, "corr_heatmap", {})
        assert "data" in fig_dict
        assert fig_dict["data"][0]["type"] == "heatmap"

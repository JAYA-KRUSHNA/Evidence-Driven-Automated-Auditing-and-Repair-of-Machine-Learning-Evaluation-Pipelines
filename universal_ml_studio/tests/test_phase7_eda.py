"""
Phase 7 — Tests for the EDA Engine.

Tests all 7 analysis modules:
    overview, statistics, missing_analysis, distribution,
    correlation, outlier_summary, target_analysis, save-to-DB
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

_proj = Path(__file__).resolve().parents[2]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

from universal_ml_studio.eda.engine import EDAEngine
from universal_ml_studio.dataset.ingestion import DatasetIngestion


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_df(rows=150):
    rng = np.random.RandomState(42)
    data = {
        "num1":   rng.randn(rows),
        "num2":   rng.randn(rows) * 10 + 50,
        "num3":   rng.exponential(5, size=rows),
        "cat1":   rng.choice(["A", "B", "C"], size=rows),
        "target": rng.choice([0, 1], size=rows, p=[0.6, 0.4]),
    }
    df = pd.DataFrame(data)
    # Some missing values
    df.loc[:4, "num1"] = np.nan
    return df


def _meta(df):
    return DatasetIngestion.extract_metadata(df)


# ═══════════════════════════════════════════════════════════════════════════════
#  Full analyze() call
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnalyze:

    def test_analyze_returns_all_modules(self):
        df = _make_df()
        result = EDAEngine.analyze(df, _meta(df), target_column="target")
        for key in ("overview", "statistics", "missing_analysis",
                    "distribution", "correlation", "outlier_summary", "target_analysis"):
            assert key in result, f"Missing key: {key}"

    def test_analyze_without_target(self):
        df = _make_df()
        result = EDAEngine.analyze(df, _meta(df), target_column=None)
        assert result["target_analysis"]["available"] is False

    def test_analyze_is_json_serialisable(self):
        import json
        df = _make_df()
        result = EDAEngine.analyze(df, _meta(df), target_column="target")
        # Should not raise
        json.dumps(result)


# ═══════════════════════════════════════════════════════════════════════════════
#  Module 1 — Overview
# ═══════════════════════════════════════════════════════════════════════════════

class TestOverview:

    def test_shape_correct(self):
        df = _make_df(rows=150)
        result = EDAEngine.analyze(df, _meta(df))
        ov = result["overview"]
        assert ov["num_rows"] == 150
        assert ov["num_columns"] == len(df.columns)

    def test_missing_count_correct(self):
        df = _make_df()
        result = EDAEngine.analyze(df, _meta(df))
        ov = result["overview"]
        assert ov["total_missing_cells"] == 5  # we set 5 NaN in num1

    def test_duplicate_rows(self):
        df = _make_df()
        result = EDAEngine.analyze(df, _meta(df))
        assert "duplicate_rows" in result["overview"]
        assert isinstance(result["overview"]["duplicate_rows"], int)


# ═══════════════════════════════════════════════════════════════════════════════
#  Module 2 — Statistics
# ═══════════════════════════════════════════════════════════════════════════════

class TestStatistics:

    def test_numerical_stats_present(self):
        df = _make_df()
        result = EDAEngine.analyze(df, _meta(df))
        stats = result["statistics"]["numerical"]
        assert "num1" in stats
        for key in ("count", "mean", "std", "min", "median", "max", "skewness"):
            assert key in stats["num1"], f"Missing stat: {key}"

    def test_categorical_stats_present(self):
        df = _make_df()
        result = EDAEngine.analyze(df, _meta(df))
        stats = result["statistics"]["categorical"]
        assert "cat1" in stats
        assert "nunique" in stats["cat1"]
        assert stats["cat1"]["nunique"] == 3

    def test_no_nan_in_output(self):
        """All stats values should be Python float or None, not NaN."""
        import math
        df = _make_df()
        result = EDAEngine.analyze(df, _meta(df))
        for col, s in result["statistics"]["numerical"].items():
            for k, v in s.items():
                if isinstance(v, float):
                    assert not math.isnan(v), f"NaN in stats[{col}][{k}]"


# ═══════════════════════════════════════════════════════════════════════════════
#  Module 3 — Missing analysis
# ═══════════════════════════════════════════════════════════════════════════════

class TestMissingAnalysis:

    def test_per_column_missing(self):
        df = _make_df()
        result = EDAEngine.analyze(df, _meta(df))
        ma = result["missing_analysis"]
        assert "num1" in ma["per_column"]
        assert ma["per_column"]["num1"]["missing_count"] == 5

    def test_columns_with_no_missing_not_listed(self):
        df = _make_df()
        result = EDAEngine.analyze(df, _meta(df))
        ma = result["missing_analysis"]
        # num2 has no missing
        assert "num2" not in ma["per_column"]

    def test_rows_with_missing(self):
        df = _make_df()
        result = EDAEngine.analyze(df, _meta(df))
        ma = result["missing_analysis"]
        assert ma["rows_with_missing"] == 5


# ═══════════════════════════════════════════════════════════════════════════════
#  Module 4 — Distribution
# ═══════════════════════════════════════════════════════════════════════════════

class TestDistribution:

    def test_histogram_present(self):
        df = _make_df()
        result = EDAEngine.analyze(df, _meta(df))
        dist = result["distribution"]["numerical"]
        assert "num2" in dist
        hist = dist["num2"]["histogram"]
        assert "counts" in hist
        assert "bin_edges" in hist
        assert len(hist["counts"]) == 10

    def test_percentiles_present(self):
        df = _make_df()
        result = EDAEngine.analyze(df, _meta(df))
        dist = result["distribution"]["numerical"]["num2"]
        for p in ("p5", "p25", "p50", "p75", "p95"):
            assert p in dist["percentiles"]

    def test_categorical_distribution(self):
        df = _make_df()
        result = EDAEngine.analyze(df, _meta(df))
        cat_dist = result["distribution"]["categorical"]
        assert "cat1" in cat_dist
        assert cat_dist["cat1"]["num_categories"] == 3


# ═══════════════════════════════════════════════════════════════════════════════
#  Module 5 — Correlation
# ═══════════════════════════════════════════════════════════════════════════════

class TestCorrelation:

    def test_correlation_available(self):
        df = _make_df()
        result = EDAEngine.analyze(df, _meta(df))
        corr = result["correlation"]
        assert corr.get("available") is True

    def test_correlation_matrix_present(self):
        df = _make_df()
        result = EDAEngine.analyze(df, _meta(df))
        corr = result["correlation"]
        assert "matrix" in corr
        # All numerical cols should be keys
        for col in ("num1", "num2", "num3"):
            assert col in corr["matrix"]

    def test_highly_correlated_pairs(self):
        """Artificially correlated columns should appear in high-pairs."""
        rng = np.random.RandomState(42)
        base = rng.randn(200)
        df = pd.DataFrame({
            "a": base,
            "b": base * 1.01 + rng.randn(200) * 0.01,  # near-perfect correlation
            "target": rng.choice([0, 1], size=200),
        })
        meta = DatasetIngestion.extract_metadata(df)
        result = EDAEngine.analyze(df, meta, target_column="target")
        corr = result["correlation"]
        assert corr["num_high_pairs"] >= 1


# ═══════════════════════════════════════════════════════════════════════════════
#  Module 6 — Outlier summary
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutlierSummary:

    def test_outlier_summary_has_per_column(self):
        df = _make_df()
        result = EDAEngine.analyze(df, _meta(df))
        ols = result["outlier_summary"]
        assert "per_column" in ols
        assert "total_iqr_outliers" in ols

    def test_extreme_values_detected(self):
        rng = np.random.RandomState(42)
        data = list(rng.randn(100)) + [999, -999]
        df = pd.DataFrame({"val": data, "target": [0, 1] * 51})
        meta = DatasetIngestion.extract_metadata(df)
        result = EDAEngine.analyze(df, meta, target_column="target")
        ols = result["outlier_summary"]
        assert ols["total_iqr_outliers"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Module 7 — Target analysis
# ═══════════════════════════════════════════════════════════════════════════════

class TestTargetAnalysis:

    def test_classification_target_analysis(self):
        df = _make_df()
        result = EDAEngine.analyze(df, _meta(df), target_column="target")
        ta = result["target_analysis"]
        assert ta["available"] is True
        assert ta["inferred_task"] == "classification"
        assert "class_distribution" in ta
        assert ta["num_classes"] == 2

    def test_regression_target_analysis(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "f1": rng.randn(200),
            "f2": rng.randn(200),
            "target": rng.randn(200) * 100,
        })
        meta = DatasetIngestion.extract_metadata(df)
        result = EDAEngine.analyze(df, meta, target_column="target")
        ta = result["target_analysis"]
        assert ta["inferred_task"] == "regression"
        assert "mean" in ta
        assert "std" in ta

    def test_imbalance_ratio_computed(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "f1": rng.randn(200),
            "target": [0] * 180 + [1] * 20,
        })
        meta = DatasetIngestion.extract_metadata(df)
        result = EDAEngine.analyze(df, meta, target_column="target")
        ta = result["target_analysis"]
        assert ta["imbalance_ratio"] == 9.0
        assert ta["is_balanced"] is False


# ═══════════════════════════════════════════════════════════════════════════════
#  Database persistence
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveEDA:

    def test_save_to_db(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from universal_ml_studio.database.models import Base, Dataset, EDAResult

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        ds = Dataset(
            id="eda_test",
            name="test_ds",
            original_name="test.csv",
            file_path="/tmp/test.csv",
            file_format="csv",
        )
        session.add(ds)
        session.commit()

        df = _make_df()
        eda_result = EDAEngine.analyze(df, _meta(df), target_column="target")
        records = EDAEngine.save(eda_result, "eda_test", session=session)
        session.commit()

        saved = session.query(EDAResult).filter_by(dataset_id="eda_test").all()
        assert len(saved) == len(eda_result)  # one row per module
        module_types = {r.analysis_type for r in saved}
        assert "overview" in module_types
        assert "statistics" in module_types

        session.close()

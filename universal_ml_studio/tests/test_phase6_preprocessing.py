"""
Phase 6 — Tests for the Preprocessing Engine.

Tests:
    - Missing value imputation (mean, median, mode, zero, drop)
    - Categorical encoding (onehot, label, frequency)
    - Feature scaling (standard, minmax, robust)
    - Outlier handling (iqr_cap, zscore_cap, iqr_remove)
    - Datetime feature extraction
    - Auto-preprocess pipeline
    - Database persistence
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

_proj = Path(__file__).resolve().parents[2]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

from universal_ml_studio.preprocessing.engine import PreprocessingEngine
from universal_ml_studio.dataset.ingestion import DatasetIngestion


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_df():
    """DataFrame with numerical, categorical, and some missing values."""
    rng = np.random.RandomState(42)
    n = 100
    data = {
        "num1": rng.randn(n),
        "num2": rng.randn(n) * 10 + 50,
        "cat1": rng.choice(["A", "B", "C"], size=n),
        "cat2": rng.choice(["X", "Y"], size=n),
        "target": rng.choice([0, 1], size=n),
    }
    df = pd.DataFrame(data)
    # Introduce missing values
    df.loc[0:4, "num1"] = np.nan
    df.loc[10:14, "cat1"] = np.nan
    return df


def _make_metadata(df):
    return DatasetIngestion.extract_metadata(df)


# ═══════════════════════════════════════════════════════════════════════════════
#  Missing value imputation
# ═══════════════════════════════════════════════════════════════════════════════

class TestImputeMissing:

    def test_impute_mean(self):
        df = _make_df()
        result, op = PreprocessingEngine.impute_missing(df, strategy="mean", columns=["num1"])
        assert result["num1"].isnull().sum() == 0
        assert op["operation"] == "impute_mean"

    def test_impute_median(self):
        df = _make_df()
        result, op = PreprocessingEngine.impute_missing(df, strategy="median", columns=["num1"])
        assert result["num1"].isnull().sum() == 0
        assert op["operation"] == "impute_median"

    def test_impute_mode(self):
        df = _make_df()
        result, op = PreprocessingEngine.impute_missing(df, strategy="mode", columns=["cat1"])
        assert result["cat1"].isnull().sum() == 0
        assert op["operation"] == "impute_mode"

    def test_impute_zero(self):
        df = _make_df()
        result, op = PreprocessingEngine.impute_missing(df, strategy="zero", columns=["num1"])
        assert result["num1"].isnull().sum() == 0
        # Check that missing values were filled with 0
        assert (result["num1"].iloc[0:5] == 0).all()

    def test_impute_drop_rows(self):
        df = _make_df()
        result, op = PreprocessingEngine.impute_missing(df, strategy="drop_rows", columns=["num1"])
        assert len(result) < len(df)
        assert result["num1"].isnull().sum() == 0

    def test_impute_preserves_non_missing(self):
        df = _make_df()
        original_valid = df["num1"].dropna().values.copy()
        result, _ = PreprocessingEngine.impute_missing(df, strategy="median", columns=["num1"])
        # Non-missing values should be unchanged
        np.testing.assert_array_equal(
            result["num1"].iloc[5:].values,
            df["num1"].iloc[5:].values,
        )

    def test_auto_detect_columns(self):
        df = _make_df()
        result, op = PreprocessingEngine.impute_missing(df, strategy="median")
        # Should auto-detect columns with missing values
        assert len(op["affected_columns"]) > 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Categorical encoding
# ═══════════════════════════════════════════════════════════════════════════════

class TestEncodeCategorical:

    def test_onehot_encoding(self):
        df = _make_df()
        # Fill NaN first for clean test
        df["cat1"] = df["cat1"].fillna("A")
        result, op = PreprocessingEngine.encode_categorical(
            df, method="onehot", columns=["cat1"],
        )
        assert "cat1" not in result.columns
        assert any(c.startswith("cat1_") for c in result.columns)
        assert op["operation"] == "encode_onehot"

    def test_label_encoding(self):
        df = _make_df()
        df["cat1"] = df["cat1"].fillna("A")
        result, op = PreprocessingEngine.encode_categorical(
            df, method="label", columns=["cat1"],
        )
        assert "cat1" in result.columns
        assert result["cat1"].dtype in (np.int32, np.int64, int)
        assert op["operation"] == "encode_label"

    def test_frequency_encoding(self):
        df = _make_df()
        df["cat1"] = df["cat1"].fillna("A")
        result, op = PreprocessingEngine.encode_categorical(
            df, method="frequency", columns=["cat1"],
        )
        assert "cat1" in result.columns
        assert result["cat1"].dtype == np.float64
        assert op["operation"] == "encode_frequency"

    def test_onehot_preserves_other_columns(self):
        df = _make_df()
        df["cat1"] = df["cat1"].fillna("A")
        result, _ = PreprocessingEngine.encode_categorical(
            df, method="onehot", columns=["cat1"],
        )
        assert "num1" in result.columns
        assert "num2" in result.columns
        assert "target" in result.columns


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature scaling
# ═══════════════════════════════════════════════════════════════════════════════

class TestScaleFeatures:

    def test_standard_scaler(self):
        df = _make_df().select_dtypes(include=["number"])
        result, op = PreprocessingEngine.scale_features(
            df, method="standard", columns=["num1", "num2"],
        )
        # Standard scaled data should have mean ≈ 0 and std ≈ 1
        assert abs(result["num2"].dropna().mean()) < 0.1
        assert abs(result["num2"].dropna().std() - 1.0) < 0.1
        assert op["operation"] == "scale_standard"

    def test_minmax_scaler(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result, op = PreprocessingEngine.scale_features(
            df, method="minmax", columns=["a"],
        )
        assert result["a"].min() >= 0
        assert result["a"].max() <= 1
        assert op["operation"] == "scale_minmax"

    def test_robust_scaler(self):
        df = _make_df().select_dtypes(include=["number"])
        result, op = PreprocessingEngine.scale_features(
            df, method="robust", columns=["num2"],
        )
        assert op["operation"] == "scale_robust"
        # Robust scaler should transform the data
        assert not np.allclose(result["num2"].dropna().values, df["num2"].dropna().values)

    def test_empty_columns_no_crash(self):
        df = _make_df()
        result, op = PreprocessingEngine.scale_features(
            df, method="standard", columns=["nonexistent_col"],
        )
        assert len(op["affected_columns"]) == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Outlier handling
# ═══════════════════════════════════════════════════════════════════════════════

class TestHandleOutliers:

    def test_iqr_cap(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({"val": list(rng.randn(97)) + [100, -100, 200]})
        result, op = PreprocessingEngine.handle_outliers(
            df, method="iqr_cap", columns=["val"],
        )
        assert result["val"].max() < 200
        assert result["val"].min() > -200
        assert op["operation"] == "outlier_iqr_cap"

    def test_zscore_cap(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({"val": list(rng.randn(97)) + [100, -100, 200]})
        result, op = PreprocessingEngine.handle_outliers(
            df, method="zscore_cap", columns=["val"],
        )
        assert result["val"].max() < 200
        assert op["operation"] == "outlier_zscore_cap"

    def test_iqr_remove(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({"val": list(rng.randn(97)) + [100, -100, 200]})
        result, op = PreprocessingEngine.handle_outliers(
            df, method="iqr_remove", columns=["val"],
        )
        assert len(result) < len(df)
        assert op["operation"] == "outlier_iqr_remove"

    def test_no_outliers_no_change(self):
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        result, op = PreprocessingEngine.handle_outliers(
            df, method="iqr_cap", columns=["val"],
        )
        pd.testing.assert_frame_equal(result, df)


# ═══════════════════════════════════════════════════════════════════════════════
#  Datetime feature extraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatetimeExtraction:

    def test_extract_creates_features(self):
        df = pd.DataFrame({
            "date_col": pd.date_range("2020-01-01", periods=10, freq="D"),
            "val": range(10),
        })
        result, op = PreprocessingEngine.extract_datetime_features(
            df, columns=["date_col"],
        )
        assert "date_col" not in result.columns
        assert "date_col_year" in result.columns
        assert "date_col_month" in result.columns
        assert "date_col_day" in result.columns
        assert "date_col_dayofweek" in result.columns

    def test_string_dates_parsed(self):
        df = pd.DataFrame({
            "date_col": ["2020-01-01", "2020-06-15", "2021-12-31"],
            "val": [1, 2, 3],
        })
        result, op = PreprocessingEngine.extract_datetime_features(
            df, columns=["date_col"],
        )
        assert "date_col_year" in result.columns


# ═══════════════════════════════════════════════════════════════════════════════
#  Auto-preprocess pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestAutoPreprocess:

    def test_auto_preprocess_runs(self):
        df = _make_df()
        meta = _make_metadata(df)
        result, ops = PreprocessingEngine.auto_preprocess(
            df, meta, target_column="target",
        )
        assert isinstance(result, pd.DataFrame)
        assert len(ops) > 0
        # Target should be preserved
        assert "target" in result.columns

    def test_auto_preprocess_no_missing(self):
        df = _make_df()
        meta = _make_metadata(df)
        result, ops = PreprocessingEngine.auto_preprocess(
            df, meta, target_column="target",
        )
        # After preprocessing, numerical columns should have no missing values
        num_missing = result.select_dtypes(include=["number"]).isnull().sum().sum()
        assert num_missing == 0

    def test_auto_preprocess_returns_operations_log(self):
        df = _make_df()
        meta = _make_metadata(df)
        result, ops = PreprocessingEngine.auto_preprocess(
            df, meta, target_column="target",
        )
        for op in ops:
            assert "operation" in op
            assert "category" in op
            assert "affected_columns" in op

    def test_auto_preprocess_with_identifiers(self):
        df = _make_df()
        df["id_col"] = range(len(df))
        meta = _make_metadata(df)
        meta["identifier_columns"] = ["id_col"]
        result, ops = PreprocessingEngine.auto_preprocess(
            df, meta, target_column="target",
        )
        # Identifier columns should be dropped
        assert "id_col" not in result.columns


# ═══════════════════════════════════════════════════════════════════════════════
#  Database persistence
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveHistory:

    def test_save_operations_to_db(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from universal_ml_studio.database.models import Base, Dataset, PreprocessingHistory

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        ds = Dataset(
            id="test123",
            name="test_ds",
            original_name="test.csv",
            file_path="/tmp/test.csv",
            file_format="csv",
        )
        session.add(ds)
        session.commit()

        operations = [
            {
                "operation": "impute_median",
                "category": "missing_values",
                "parameters": {"strategy": "median"},
                "affected_columns": ["col1", "col2"],
                "rows_before": 100,
                "rows_after": 100,
            },
            {
                "operation": "encode_onehot",
                "category": "encoding",
                "parameters": {"method": "onehot"},
                "affected_columns": ["cat1"],
                "rows_before": 100,
                "rows_after": 100,
            },
        ]

        records = PreprocessingEngine.save_history(operations, "test123", session=session)
        session.commit()

        assert len(records) == 2
        saved = session.query(PreprocessingHistory).filter_by(dataset_id="test123").all()
        assert len(saved) == 2
        assert saved[0].operation == "impute_median"
        assert saved[1].operation == "encode_onehot"

        session.close()

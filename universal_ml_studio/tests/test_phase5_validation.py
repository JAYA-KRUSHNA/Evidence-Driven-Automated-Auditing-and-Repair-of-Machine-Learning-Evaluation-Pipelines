"""
Phase 5 — Tests for the Validation Engine.

Tests all 4 tiers:
    Tier 1: File-level checks (format, size, readability)
    Tier 2: Structural checks (rows, columns, duplicates, constants)
    Tier 3: Data quality checks (missing, outliers, cardinality, types)
    Tier 4: ML readiness checks (target, balance, leakage, features)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
_proj = Path(__file__).resolve().parents[2]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

from universal_ml_studio.validation.engine import (
    ValidationEngine,
    ValidationResult,
    CheckResult,
)
from universal_ml_studio.dataset.ingestion import DatasetIngestion


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_clean_df(rows=100, cols=5):
    """Create a clean DataFrame with no issues."""
    rng = np.random.RandomState(42)
    data = {f"feature_{i}": rng.randn(rows) for i in range(cols)}
    data["target"] = rng.choice([0, 1], size=rows, p=[0.5, 0.5])
    return pd.DataFrame(data)


def _make_metadata(df, file_format="csv", file_size_bytes=1024):
    """Create minimal metadata dict."""
    meta = DatasetIngestion.extract_metadata(df)
    meta["file_format"] = file_format
    meta["file_size_bytes"] = file_size_bytes
    return meta


def _find_check(result: ValidationResult, check_name: str) -> CheckResult | None:
    """Find a check by name in the result."""
    for c in result.checks:
        if c.check_name == check_name:
            return c
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Tier 1 — File-level
# ═══════════════════════════════════════════════════════════════════════════════

class TestTier1FileChecks:

    def test_supported_format_passes(self):
        df = _make_clean_df()
        meta = _make_metadata(df, file_format="csv")
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "file_format_supported")
        assert check is not None
        assert check.status == "passed"

    def test_unsupported_format_fails(self):
        df = _make_clean_df()
        meta = _make_metadata(df, file_format="json")
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "file_format_supported")
        assert check is not None
        assert check.status == "failed"

    def test_file_size_within_limit(self):
        df = _make_clean_df()
        meta = _make_metadata(df, file_size_bytes=1024 * 1024 * 10)  # 10 MB
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "file_size_within_limit")
        assert check is not None
        assert check.status == "passed"

    def test_file_size_exceeds_limit(self):
        df = _make_clean_df()
        meta = _make_metadata(df, file_size_bytes=1024 * 1024 * 300)  # 300 MB
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "file_size_within_limit")
        assert check is not None
        assert check.status == "failed"

    def test_file_readable_always_passes(self):
        df = _make_clean_df()
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "file_readable")
        assert check is not None
        assert check.status == "passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  Tier 2 — Structural
# ═══════════════════════════════════════════════════════════════════════════════

class TestTier2StructuralChecks:

    def test_minimum_row_count_passes(self):
        df = _make_clean_df(rows=100)
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "minimum_row_count")
        assert check is not None
        assert check.status == "passed"

    def test_minimum_row_count_fails(self):
        df = _make_clean_df(rows=5)
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "minimum_row_count")
        assert check is not None
        assert check.status == "failed"

    def test_minimum_column_count_passes(self):
        df = _make_clean_df(cols=3)
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "minimum_column_count")
        assert check is not None
        assert check.status == "passed"

    def test_minimum_column_count_fails(self):
        df = pd.DataFrame({"only_col": [1, 2, 3]})
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta)
        check = _find_check(result, "minimum_column_count")
        assert check is not None
        assert check.status == "failed"

    def test_no_duplicate_column_names(self):
        df = _make_clean_df()
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "no_duplicate_column_names")
        assert check is not None
        assert check.status == "passed"

    def test_empty_columns_detected(self):
        df = _make_clean_df()
        df["empty_col"] = np.nan
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "no_empty_columns")
        assert check is not None
        assert check.status == "warning"
        assert "empty_col" in check.affected_columns

    def test_constant_column_detected(self):
        df = _make_clean_df()
        df["const"] = 42
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "no_constant_columns")
        assert check is not None
        assert check.status == "warning"
        assert "const" in check.affected_columns

    def test_clean_column_names(self):
        df = _make_clean_df()
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "column_name_hygiene")
        assert check is not None
        assert check.status == "passed"

    def test_no_duplicate_rows_clean(self):
        df = _make_clean_df()
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "no_duplicate_rows")
        assert check is not None
        # Could be passed or warning depending on random data
        assert check.status in ("passed", "warning")


# ═══════════════════════════════════════════════════════════════════════════════
#  Tier 3 — Data Quality
# ═══════════════════════════════════════════════════════════════════════════════

class TestTier3QualityChecks:

    def test_no_missing_values(self):
        df = _make_clean_df()
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "overall_missing_values")
        assert check is not None
        assert check.status == "passed"

    def test_high_missing_values_warning(self):
        df = _make_clean_df(rows=100)
        # Introduce 15% missing in a feature column
        df.loc[:14, "feature_0"] = np.nan
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "per_column_missing_values")
        assert check is not None
        assert check.status == "warning"

    def test_massive_missing_fails(self):
        """A dataset with >80% overall missing values should fail."""
        # 5 feature columns each 90% NaN + 1 target column = ~75% overall missing
        # Need >80% total → make 9 columns 95% NaN + 1 target
        n = 100
        data = {f"f{i}": [np.nan] * 95 + [1.0] * 5 for i in range(9)}
        data["target"] = [0] * 50 + [1] * 50
        df = pd.DataFrame(data)
        # overall missing = 9*95 / (10*100) = 855/1000 = 85.5%
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "overall_missing_values")
        assert check is not None
        assert check.status == "failed"

    def test_outlier_detection_clean(self):
        df = _make_clean_df()
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "outlier_detection")
        assert check is not None
        # Clean normally-distributed data should pass
        assert check.status == "passed"

    def test_high_cardinality_warning(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "cat_col": [f"val_{i}" for i in range(200)],
            "num_col": rng.randn(200),
            "target": rng.choice([0, 1], size=200),
        })
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "high_cardinality")
        assert check is not None
        assert check.status == "warning"

    def test_no_infinite_values_clean(self):
        df = _make_clean_df()
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "no_infinite_values")
        assert check is not None
        assert check.status == "passed"

    def test_infinite_values_detected(self):
        df = _make_clean_df()
        df.loc[0, "feature_0"] = np.inf
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "no_infinite_values")
        assert check is not None
        assert check.status == "warning"

    def test_mixed_types_detected(self):
        df = pd.DataFrame({
            "mixed": ["hello", "world", "123", "456", "foo"] * 20,
            "num": list(range(100)),
            "target": [0, 1] * 50,
        })
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "no_mixed_types")
        assert check is not None
        # Should warn because "mixed" has both parsable numbers and strings
        assert check.status == "warning"

    def test_no_whitespace_strings_clean(self):
        df = _make_clean_df()
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "no_whitespace_only_strings")
        assert check is not None
        assert check.status == "passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  Tier 4 — ML Readiness
# ═══════════════════════════════════════════════════════════════════════════════

class TestTier4MLReadiness:

    def test_target_column_exists(self):
        df = _make_clean_df()
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "target_column_exists")
        assert check is not None
        assert check.status == "passed"

    def test_target_column_missing(self):
        df = _make_clean_df()
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="nonexistent")
        check = _find_check(result, "target_column_exists")
        assert check is not None
        assert check.status == "failed"

    def test_no_target_specified_warning(self):
        df = _make_clean_df()
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column=None)
        check = _find_check(result, "target_column_specified")
        assert check is not None
        assert check.status == "warning"

    def test_target_no_missing_values(self):
        df = _make_clean_df()
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "target_no_missing")
        assert check is not None
        assert check.status == "passed"

    def test_sufficient_features(self):
        df = _make_clean_df(cols=5)
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "sufficient_features")
        assert check is not None
        assert check.status == "passed"

    def test_task_type_classification_inferred(self):
        df = _make_clean_df()
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "task_type_inference")
        assert check is not None
        assert check.status == "passed"
        assert check.details["inferred_task"] == "classification"

    def test_task_type_regression_inferred(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "f1": rng.randn(100),
            "f2": rng.randn(100),
            "target": rng.randn(100) * 100,  # continuous target
        })
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "task_type_inference")
        assert check is not None
        assert check.details["inferred_task"] == "regression"

    def test_class_balance_ok(self):
        df = _make_clean_df()  # 50/50 split
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "class_balance")
        assert check is not None
        assert check.status == "passed"

    def test_class_imbalance_warning(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "f1": rng.randn(200),
            "f2": rng.randn(200),
            "target": [0] * 190 + [1] * 10,  # 19:1 ratio
        })
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "class_balance")
        assert check is not None
        assert check.status == "warning"

    def test_min_samples_per_class_ok(self):
        df = _make_clean_df()
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        check = _find_check(result, "min_samples_per_class")
        assert check is not None
        assert check.status == "passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  Integration — Overall result
# ═══════════════════════════════════════════════════════════════════════════════

class TestOverallResult:

    def test_clean_dataset_is_valid(self):
        df = _make_clean_df()
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        assert result.status == "valid"
        assert result.is_trainable is True
        assert result.failed_checks == 0

    def test_dataset_with_issues_gets_warnings(self):
        df = _make_clean_df()
        df["empty_col"] = np.nan
        df["const_col"] = 99
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        assert result.status == "valid_with_warnings"
        assert result.is_trainable is True
        assert result.warning_count > 0

    def test_invalid_dataset(self):
        """Tiny dataset with bad format should be invalid."""
        df = pd.DataFrame({"a": [1], "target": [0]})
        meta = _make_metadata(df, file_format="json")
        result = ValidationEngine.validate(df, meta, target_column="target")
        assert result.status == "invalid"
        assert result.is_trainable is False

    def test_summary_has_categories(self):
        df = _make_clean_df()
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        assert "categories" in result.summary
        for cat in ("file", "structural", "quality", "ml_readiness"):
            assert cat in result.summary["categories"]

    def test_total_checks_count(self):
        df = _make_clean_df()
        meta = _make_metadata(df)
        result = ValidationEngine.validate(df, meta, target_column="target")
        assert result.total_checks >= 20  # we expect 20+ checks for a full run
        assert result.total_checks == result.passed_checks + result.warning_count + result.failed_checks


# ═══════════════════════════════════════════════════════════════════════════════
#  Database persistence
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveReport:

    def test_save_report_to_db(self, tmp_path):
        """Test that save_report persists to the database."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from universal_ml_studio.database.models import Base, Dataset, ValidationReport, ValidationCheck

        # Create in-memory database
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Create a dummy dataset record
        ds = Dataset(
            id="test123",
            name="test_ds",
            original_name="test.csv",
            file_path="/tmp/test.csv",
            file_format="csv",
        )
        session.add(ds)
        session.commit()

        # Validate
        df = _make_clean_df()
        meta = _make_metadata(df)
        validation_result = ValidationEngine.validate(df, meta, target_column="target")

        # Save
        report = ValidationEngine.save_report(validation_result, "test123", session=session)
        session.commit()

        # Verify
        assert report.id is not None
        assert report.dataset_id == "test123"
        assert report.status == validation_result.status
        assert report.total_checks == validation_result.total_checks

        # Verify checks were saved
        checks = session.query(ValidationCheck).filter_by(report_id=report.id).all()
        assert len(checks) == validation_result.total_checks

        session.close()

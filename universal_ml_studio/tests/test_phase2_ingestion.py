"""
Phase 2 — Tests for dataset/ingestion.py

Covers:
    - CSV reading
    - Excel reading (.xlsx)
    - Metadata extraction (column types, numerical/categorical, identifiers)
    - Datetime detection
    - Error cases (missing file, empty file, unsupported format)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from universal_ml_studio.dataset.ingestion import DatasetIngestion, DatasetReadError


# ═══════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_csv(tmp_path):
    """Write a small CSV and return its path."""
    df = pd.DataFrame({
        "id": range(1, 51),
        "name": [f"item_{i}" for i in range(1, 51)],
        "price": np.random.uniform(5, 100, 50).round(2),
        "quantity": np.random.randint(1, 500, 50),
        "category": np.random.choice(["A", "B", "C"], 50),
        "in_stock": np.random.choice([True, False], 50),
    })
    path = tmp_path / "products.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_xlsx(tmp_path):
    """Write a small Excel .xlsx and return its path."""
    df = pd.DataFrame({
        "student_id": range(1, 31),
        "score": np.random.randint(40, 100, 30),
        "grade": np.random.choice(["A", "B", "C", "D", "F"], 30),
    })
    path = tmp_path / "students.xlsx"
    df.to_excel(path, index=False, engine="openpyxl")
    return path


@pytest.fixture
def csv_with_missing(tmp_path):
    """CSV with missing values."""
    df = pd.DataFrame({
        "a": [1, 2, None, 4, 5],
        "b": ["x", None, "z", None, "w"],
        "c": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    path = tmp_path / "missing.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def csv_with_dates(tmp_path):
    """CSV with a date column stored as object."""
    df = pd.DataFrame({
        "order_id": range(1, 21),
        "order_date": pd.date_range("2024-01-01", periods=20).astype(str),
        "amount": np.random.uniform(10, 500, 20).round(2),
    })
    path = tmp_path / "orders.csv"
    df.to_csv(path, index=False)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  Read tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRead:

    def test_read_csv(self, sample_csv):
        df, meta = DatasetIngestion.read(sample_csv)
        assert len(df) == 50
        assert meta["num_rows"] == 50
        assert meta["num_columns"] == 6
        assert meta["file_format"] == "csv"

    def test_read_xlsx(self, sample_xlsx):
        df, meta = DatasetIngestion.read(sample_xlsx)
        assert len(df) == 30
        assert meta["file_format"] == "xlsx"

    def test_file_not_found(self):
        with pytest.raises(DatasetReadError, match="File not found"):
            DatasetIngestion.read("/nonexistent/file.csv")

    def test_unsupported_format(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text('{"a": 1}')
        with pytest.raises(DatasetReadError, match="Unsupported format"):
            DatasetIngestion.read(p)

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("")
        with pytest.raises(DatasetReadError, match="empty"):
            DatasetIngestion.read(p)


# ═══════════════════════════════════════════════════════════════════════════════
#  Metadata tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetadata:

    def test_column_classification(self, sample_csv):
        df, meta = DatasetIngestion.read(sample_csv)
        assert "price" in meta["numerical_columns"]
        assert "quantity" in meta["numerical_columns"]
        assert "category" in meta["categorical_columns"]

    def test_missing_value_detection(self, csv_with_missing):
        df, meta = DatasetIngestion.read(csv_with_missing)
        assert meta["total_missing_cells"] > 0
        assert "a" in meta["missing_values"] or "b" in meta["missing_values"]

    def test_duplicate_count(self, tmp_path):
        df = pd.DataFrame({"x": [1, 1, 2, 2, 3], "y": [10, 10, 20, 20, 30]})
        path = tmp_path / "dupes.csv"
        df.to_csv(path, index=False)
        _, meta = DatasetIngestion.read(path)
        assert meta["duplicate_rows"] == 2

    def test_identifier_detection(self, sample_csv):
        _, meta = DatasetIngestion.read(sample_csv)
        assert "id" in meta["identifier_columns"]

    def test_datetime_detection(self, csv_with_dates):
        _, meta = DatasetIngestion.read(csv_with_dates)
        assert "order_date" in meta["datetime_columns"]

    def test_memory_usage_positive(self, sample_csv):
        _, meta = DatasetIngestion.read(sample_csv)
        assert meta["memory_bytes"] > 0

    def test_extract_metadata_standalone(self):
        """extract_metadata works on any DataFrame without a file path."""
        df = pd.DataFrame({"num": [1, 2, 3], "cat": ["a", "b", "c"]})
        meta = DatasetIngestion.extract_metadata(df)
        assert meta["num_rows"] == 3
        assert "file_name" not in meta
        assert "num" in meta["numerical_columns"]
        assert "cat" in meta["categorical_columns"]

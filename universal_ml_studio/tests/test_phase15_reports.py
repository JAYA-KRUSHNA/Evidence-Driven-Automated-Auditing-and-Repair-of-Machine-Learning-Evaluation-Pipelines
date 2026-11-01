"""
Unit tests for Phase 15: Report Generator.
"""

from pathlib import Path
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from universal_ml_studio.database.connection import init_db
from universal_ml_studio.dataset.registry import DatasetRegistry
from universal_ml_studio.eda.engine import EDAEngine
from universal_ml_studio.experiment.tracker import ExperimentTracker
from universal_ml_studio.ml.classification import ClassificationEngine
from universal_ml_studio.reports.generator import ReportGenerator, ReportNotFoundError
from universal_ml_studio.validation.engine import ValidationEngine


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


@pytest.fixture
def sample_dataset():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    meta = {
        "file_name": "iris_reports_test.csv",
        "file_format": "csv",
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "column_names": list(df.columns),
        "column_types": {c: str(t) for c, t in df.dtypes.items()},
        "numerical_cols": list(iris.feature_names),
        "categorical_cols": [],
        "datetime_cols": [],
        "identifier_cols": [],
        "file_size_bytes": 1000,
    }
    registered = DatasetRegistry.register(df, meta, name="Iris Reports Test")
    # Run validation and EDA to populate DB records
    val_res = ValidationEngine.validate(df, meta)
    ValidationEngine.save_report(val_res, registered["id"])
    eda_res = EDAEngine.analyze(df, meta)
    EDAEngine.save(eda_res, registered["id"])

    yield registered
    try:
        DatasetRegistry.delete(registered["id"])
    except Exception:
        pass


class TestReportGenerator:
    def test_validation_reports_all_formats(self, sample_dataset):
        for fmt in ("json", "csv", "pdf"):
            rep = ReportGenerator.generate_validation_report(sample_dataset["id"], format=fmt)
            assert rep["id"] is not None
            assert rep["format"] == fmt
            p = Path(rep["file_path"])
            assert p.exists()
            assert p.stat().st_size > 0
            ReportGenerator.delete(rep["id"])

    def test_eda_reports_all_formats(self, sample_dataset):
        for fmt in ("json", "csv", "pdf"):
            rep = ReportGenerator.generate_eda_report(sample_dataset["id"], format=fmt)
            assert rep["id"] is not None
            assert rep["format"] == fmt
            p = Path(rep["file_path"])
            assert p.exists()
            assert p.stat().st_size > 0
            ReportGenerator.delete(rep["id"])

    def test_experiment_report(self, sample_dataset):
        exp = ExperimentTracker.create(
            dataset_id=sample_dataset["id"],
            task_type="classification",
            target_column="target",
            algorithm="random_forest",
        )
        df = DatasetRegistry.load_dataframe(sample_dataset["id"])
        train_res = ClassificationEngine.train(df, "target", "random_forest", config={"cv_folds": 2})
        ExperimentTracker.save_results(exp["id"], train_res)

        for fmt in ("json", "csv", "pdf"):
            rep = ReportGenerator.generate_experiment_report(exp["id"], format=fmt)
            assert rep["id"] is not None
            assert rep["format"] == fmt
            p = Path(rep["file_path"])
            assert p.exists()
            assert p.stat().st_size > 0
            ReportGenerator.delete(rep["id"])

        ExperimentTracker.delete(exp["id"])

    def test_full_report(self, sample_dataset):
        for fmt in ("json", "csv", "pdf"):
            rep = ReportGenerator.generate_full_report(sample_dataset["id"], format=fmt)
            assert rep["id"] is not None
            p = Path(rep["file_path"])
            assert p.exists()
            ReportGenerator.delete(rep["id"])

    def test_get_and_list_reports(self, sample_dataset):
        rep = ReportGenerator.generate_validation_report(sample_dataset["id"], format="json")
        fetched = ReportGenerator.get(rep["id"])
        assert fetched["id"] == rep["id"]

        all_reports = ReportGenerator.list_all(dataset_id=sample_dataset["id"])
        ids = [r["id"] for r in all_reports]
        assert rep["id"] in ids

        ReportGenerator.delete(rep["id"])

    def test_invalid_format_raises(self, sample_dataset):
        with pytest.raises(ValueError):
            ReportGenerator.generate_validation_report(sample_dataset["id"], format="docx")

    def test_get_nonexistent_raises(self):
        with pytest.raises(ReportNotFoundError):
            ReportGenerator.get("nonexistent_rep_id")

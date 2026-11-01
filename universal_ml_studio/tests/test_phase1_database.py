"""
Phase 1 — Tests for config.py and database/.

Verifies:
    - Config constants are sane.
    - Storage directories are created.
    - Database initialises and all 13 tables exist.
    - CRUD operations work on core tables.
    - Relationships and cascading deletes behave correctly.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import inspect, text

# ── Ensure project root is on sys.path ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from universal_ml_studio.config import (
    PROJECT_ROOT as CFG_ROOT,
    STORAGE_DIR, DATASETS_DIR, MODELS_DIR, REPORTS_DIR, SAMPLE_DATA_DIR,
    DATABASE_URL, SUPPORTED_EXTENSIONS,
    CLASSIFICATION_ALGORITHMS, REGRESSION_ALGORITHMS,
)
from universal_ml_studio.database.connection import get_engine, get_session, init_db
from universal_ml_studio.database.models import (
    Base, Dataset, DatasetVersion, ValidationReport, ValidationCheck,
    PreprocessingHistory, EDAResult, Experiment, TrainedModel, ModelFile,
    Prediction, EvaluationResult, ComparisonRun, Report,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Config tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfig:
    """Smoke tests for config.py constants."""

    def test_project_root_exists(self):
        assert CFG_ROOT.exists()

    def test_storage_dirs_exist(self):
        for d in (STORAGE_DIR, DATASETS_DIR, MODELS_DIR, REPORTS_DIR, SAMPLE_DATA_DIR):
            assert d.exists(), f"{d} was not created"

    def test_supported_extensions(self):
        assert ".csv" in SUPPORTED_EXTENSIONS
        assert ".xlsx" in SUPPORTED_EXTENSIONS
        assert ".xls" in SUPPORTED_EXTENSIONS

    def test_algorithm_lists_not_empty(self):
        assert len(CLASSIFICATION_ALGORITHMS) == 9
        assert len(REGRESSION_ALGORITHMS) == 10

    def test_database_url_format(self):
        assert DATABASE_URL.startswith("sqlite:///")


# ═══════════════════════════════════════════════════════════════════════════════
#  Database tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatabase:
    """Tests for database initialisation and ORM models."""

    @pytest.fixture(autouse=True)
    def setup_db(self, tmp_path):
        """Create a fresh in-memory database for each test."""
        import universal_ml_studio.database.connection as conn_mod
        import universal_ml_studio.config as cfg_mod

        # Override database URL to use in-memory SQLite
        self._original_url = cfg_mod.DATABASE_URL
        cfg_mod.DATABASE_URL = "sqlite:///:memory:"

        # Reset singletons so we get a fresh engine
        conn_mod._engine = None
        conn_mod._SessionFactory = None

        # Monkey-patch the module-level DATABASE_URL used by get_engine
        original_get_engine = conn_mod.get_engine

        from sqlalchemy import create_engine, event
        from sqlalchemy.orm import sessionmaker

        engine = create_engine("sqlite:///:memory:", echo=False)
        event.listen(engine, "connect", conn_mod._set_sqlite_pragma)
        Base.metadata.create_all(bind=engine)

        session_factory = sessionmaker(bind=engine)

        conn_mod._engine = engine
        conn_mod._SessionFactory = session_factory

        self.engine = engine
        self.session = session_factory()

        yield

        self.session.close()
        conn_mod._engine = None
        conn_mod._SessionFactory = None
        cfg_mod.DATABASE_URL = self._original_url

    # ── Table existence ────────────────────────────────────────────────────

    def test_all_13_tables_created(self):
        inspector = inspect(self.engine)
        table_names = set(inspector.get_table_names())
        expected = {
            "datasets", "dataset_versions", "validation_reports",
            "validation_checks", "preprocessing_history", "eda_results",
            "experiments", "trained_models", "model_files",
            "predictions", "evaluation_results", "comparison_runs", "reports",
        }
        assert expected.issubset(table_names), f"Missing tables: {expected - table_names}"

    # ── Dataset CRUD ──────────────────────────────────────────────────────

    def test_create_dataset(self):
        ds = Dataset(
            name="test_dataset",
            original_name="test.csv",
            file_path="/tmp/test.csv",
            file_format="csv",
            source="upload",
            num_rows=100,
            num_columns=5,
            column_names=["a", "b", "c", "d", "target"],
        )
        self.session.add(ds)
        self.session.commit()

        fetched = self.session.query(Dataset).filter_by(name="test_dataset").first()
        assert fetched is not None
        assert fetched.num_rows == 100
        assert fetched.file_format == "csv"
        assert len(fetched.column_names) == 5

    def test_dataset_has_auto_id(self):
        ds = Dataset(
            name="auto_id_test",
            original_name="x.csv",
            file_path="/tmp/x.csv",
            file_format="csv",
        )
        self.session.add(ds)
        self.session.commit()
        assert ds.id is not None
        assert len(ds.id) == 12

    # ── DatasetVersion ────────────────────────────────────────────────────

    def test_dataset_version_relationship(self):
        ds = Dataset(name="v_test", original_name="v.csv", file_path="/v.csv", file_format="csv")
        self.session.add(ds)
        self.session.commit()

        v1 = DatasetVersion(dataset_id=ds.id, version=1, file_path="/v1.csv", num_rows=100)
        v2 = DatasetVersion(dataset_id=ds.id, version=2, file_path="/v2.csv", num_rows=90)
        self.session.add_all([v1, v2])
        self.session.commit()

        assert len(ds.versions) == 2
        assert ds.versions[0].version in (1, 2)

    # ── Validation ────────────────────────────────────────────────────────

    def test_validation_report_with_checks(self):
        ds = Dataset(name="val_test", original_name="v.csv", file_path="/v.csv", file_format="csv")
        self.session.add(ds)
        self.session.commit()

        report = ValidationReport(
            dataset_id=ds.id,
            status="valid_with_warnings",
            total_checks=10,
            passed_checks=8,
            failed_checks=0,
            warning_count=2,
            is_trainable=True,
        )
        self.session.add(report)
        self.session.commit()

        check = ValidationCheck(
            report_id=report.id,
            category="quality",
            check_name="missing_values",
            status="warning",
            message="Column 'age' has 3% missing values",
            affected_columns=["age"],
            affected_rows=3,
            suggestion="Impute with median",
        )
        self.session.add(check)
        self.session.commit()

        assert len(report.checks) == 1
        assert report.checks[0].category == "quality"

    # ── Experiment → Model → Evaluation chain ─────────────────────────────

    def test_experiment_model_chain(self):
        ds = Dataset(name="exp_test", original_name="e.csv", file_path="/e.csv", file_format="csv")
        self.session.add(ds)
        self.session.commit()

        exp = Experiment(
            dataset_id=ds.id,
            task_type="classification",
            target_column="target",
            algorithm="random_forest",
            status="completed",
        )
        self.session.add(exp)
        self.session.commit()

        model = TrainedModel(
            experiment_id=exp.id,
            model_name="RandomForest_v1",
            algorithm="random_forest",
            task_type="classification",
            metrics={"accuracy": 0.95, "f1": 0.93},
            is_best=True,
        )
        self.session.add(model)
        self.session.commit()

        ev = EvaluationResult(
            experiment_id=exp.id,
            model_id=model.id,
            metric_name="accuracy",
            metric_value=0.95,
        )
        self.session.add(ev)
        self.session.commit()

        assert len(exp.models) == 1
        assert exp.models[0].is_best is True
        assert len(exp.evaluations) == 1

    # ── Cascade delete ────────────────────────────────────────────────────

    def test_cascade_delete_dataset(self):
        ds = Dataset(name="cascade_test", original_name="c.csv", file_path="/c.csv", file_format="csv")
        self.session.add(ds)
        self.session.commit()

        exp = Experiment(
            dataset_id=ds.id, task_type="regression",
            target_column="price", algorithm="linear_regression",
        )
        self.session.add(exp)
        self.session.commit()

        self.session.delete(ds)
        self.session.commit()

        assert self.session.query(Experiment).count() == 0

    # ── Comparison run ────────────────────────────────────────────────────

    def test_comparison_run(self):
        cr = ComparisonRun(
            dataset_id="abc123",
            task_type="classification",
            target_column="target",
            algorithms=["random_forest", "svm", "knn"],
            ranking_metric="accuracy",
            results=[
                {"algorithm": "random_forest", "accuracy": 0.95},
                {"algorithm": "svm", "accuracy": 0.92},
            ],
            best_algorithm="random_forest",
        )
        self.session.add(cr)
        self.session.commit()
        assert len(cr.algorithms) == 3

    # ── Report ────────────────────────────────────────────────────────────

    def test_report_creation(self):
        r = Report(
            report_type="full",
            format="pdf",
            file_path="/reports/report_abc.pdf",
            file_size=12345,
        )
        self.session.add(r)
        self.session.commit()
        assert r.id is not None
        assert r.format == "pdf"

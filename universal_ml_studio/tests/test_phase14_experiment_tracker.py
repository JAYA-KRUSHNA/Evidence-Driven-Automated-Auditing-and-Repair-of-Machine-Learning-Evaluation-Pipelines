"""
Unit tests for Phase 14: Experiment Tracker.
"""

import pandas as pd
import pytest
from sklearn.datasets import load_iris

from universal_ml_studio.database.connection import init_db
from universal_ml_studio.dataset.registry import DatasetRegistry
from universal_ml_studio.experiment.tracker import ExperimentTracker, ExperimentNotFoundError
from universal_ml_studio.ml.classification import ClassificationEngine


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


@pytest.fixture
def sample_dataset():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    meta = {
        "file_name": "iris_tracker_test.csv",
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
    registered = DatasetRegistry.register(df, meta, name="Iris Tracker Test")
    yield registered
    try:
        DatasetRegistry.delete(registered["id"])
    except Exception:
        pass


class TestExperimentTrackerLifecycle:
    def test_create_and_update_status(self, sample_dataset):
        exp = ExperimentTracker.create(
            dataset_id=sample_dataset["id"],
            task_type="classification",
            target_column="target",
            algorithm="random_forest",
            feature_columns=["sepal length (cm)", "sepal width (cm)"],
        )

        assert exp["id"] is not None
        assert exp["status"] == "pending"
        assert exp["algorithm"] == "random_forest"

        # Update status to running
        updated = ExperimentTracker.update_status(exp["id"], status="running")
        assert updated["status"] == "running"

        # Update status to failed
        failed = ExperimentTracker.update_status(
            exp["id"],
            status="failed",
            error_message="Simulated run failure",
        )
        assert failed["status"] == "failed"
        assert failed["error_message"] == "Simulated run failure"
        assert failed["completed_at"] is not None

        # Clean up
        ExperimentTracker.delete(exp["id"])

    def test_save_results_and_get(self, sample_dataset):
        exp = ExperimentTracker.create(
            dataset_id=sample_dataset["id"],
            task_type="classification",
            target_column="target",
            algorithm="random_forest",
            feature_columns=["sepal length (cm)", "sepal width (cm)"],
        )

        df = DatasetRegistry.load_dataframe(sample_dataset["id"])
        train_res = ClassificationEngine.train(
            df=df,
            target_column="target",
            algorithm="random_forest",
            config={"cv_folds": 3},
        )

        completed_exp = ExperimentTracker.save_results(
            experiment_id=exp["id"],
            training_result=train_res,
            is_best=True,
        )

        assert completed_exp["status"] == "completed"
        assert completed_exp["training_time_sec"] > 0

        # Retrieve full details
        full_details = ExperimentTracker.get(exp["id"])
        assert len(full_details["models"]) == 1
        assert full_details["models"][0]["is_best"] is True
        assert len(full_details["evaluations"]) >= 1

        # Check metrics in evaluations
        eval_metrics = [ev["metric_name"] for ev in full_details["evaluations"]]
        assert "accuracy" in eval_metrics

        # Clean up
        ExperimentTracker.delete(exp["id"])

    def test_log_prediction(self, sample_dataset):
        exp = ExperimentTracker.create(
            dataset_id=sample_dataset["id"],
            task_type="classification",
            target_column="target",
            algorithm="decision_tree",
        )

        pred_res = ExperimentTracker.log_prediction(
            experiment_id=exp["id"],
            input_data={"sepal length (cm)": 5.1, "sepal width (cm)": 3.5},
            prediction=0,
            confidence=0.95,
        )

        assert pred_res["id"] is not None
        assert pred_res["prediction"] == 0
        assert pred_res["confidence"] == 0.95

        details = ExperimentTracker.get(exp["id"])
        assert details["predictions_count"] == 1

        ExperimentTracker.delete(exp["id"])


class TestExperimentComparisonAndListing:
    def test_list_and_compare_experiments(self, sample_dataset):
        exp1 = ExperimentTracker.create(
            dataset_id=sample_dataset["id"],
            task_type="classification",
            target_column="target",
            algorithm="random_forest",
        )
        exp2 = ExperimentTracker.create(
            dataset_id=sample_dataset["id"],
            task_type="classification",
            target_column="target",
            algorithm="logistic_regression",
        )

        df = DatasetRegistry.load_dataframe(sample_dataset["id"])
        res1 = ClassificationEngine.train(df, "target", "random_forest", config={"cv_folds": 2})
        res2 = ClassificationEngine.train(df, "target", "logistic_regression", config={"cv_folds": 2})

        ExperimentTracker.save_results(exp1["id"], res1)
        ExperimentTracker.save_results(exp2["id"], res2)

        # List all
        all_exps = ExperimentTracker.list_all(dataset_id=sample_dataset["id"])
        exp_ids = [e["id"] for e in all_exps]
        assert exp1["id"] in exp_ids
        assert exp2["id"] in exp_ids

        # Compare
        comp = ExperimentTracker.compare([exp1["id"], exp2["id"]])
        assert comp["total_compared"] == 2
        assert len(comp["experiments"]) == 2
        assert "accuracy" in comp["compared_metrics"]

        # Clean up
        ExperimentTracker.delete(exp1["id"])
        ExperimentTracker.delete(exp2["id"])

    def test_get_nonexistent_raises(self):
        with pytest.raises(ExperimentNotFoundError):
            ExperimentTracker.get("unknown_exp_id")

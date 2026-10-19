"""Tests for experiment runner and corrections."""

import pytest
import pandas as pd
import numpy as np

from src.experiments.experiment_runner import ExperimentRunner, ExperimentConfig
from src.experiments.pipeline_builder import PipelineBuilder
from src.experiments.split_manager import SplitManager
from src.experiments.metrics import MetricsCalculator


class TestExperimentRunner:

    def test_runs_classification_experiment(self, clean_classification_df):
        """Full experiment should run and produce real metrics."""
        config = ExperimentConfig(
            dataset_id="test_001",
            target_column="target",
            task_type="classification",
            model_names=["logistic_regression"],
            random_seed=42,
            cv_folds=3,
            experiment_name="test_classification",
        )

        runner = ExperimentRunner()
        result = runner.run(clean_classification_df, config)

        assert result.status == "completed", f"Experiment should complete, got: {result.status} ({result.error})"
        assert "logistic_regression" in result.training_results
        tr = result.training_results["logistic_regression"]
        assert "accuracy" in tr.test_metrics, "Should have accuracy metric"
        assert 0 <= tr.test_metrics["accuracy"] <= 1, "Accuracy must be between 0 and 1"
        assert tr.training_time > 0, "Training time must be positive"

    def test_runs_regression_experiment(self, regression_df):
        """Regression experiment should produce real metrics."""
        config = ExperimentConfig(
            dataset_id="test_002",
            target_column="target",
            task_type="regression",
            model_names=["linear_regression"],
            random_seed=42,
            cv_folds=3,
        )

        runner = ExperimentRunner()
        result = runner.run(regression_df, config)

        assert result.status == "completed"
        tr = result.training_results["linear_regression"]
        assert "r2" in tr.test_metrics
        assert "rmse" in tr.test_metrics

    def test_invalid_target_column_fails(self, clean_classification_df):
        """Invalid target column should fail gracefully."""
        config = ExperimentConfig(
            dataset_id="test_003",
            target_column="nonexistent_column",
            task_type="classification",
        )

        runner = ExperimentRunner()
        result = runner.run(clean_classification_df, config)
        assert result.status == "failed"
        assert result.error is not None


class TestPipelineBuilder:

    def test_builds_classification_pipeline(self, clean_classification_df):
        """Should build a working pipeline."""
        X = clean_classification_df.drop(columns=["target"])
        pipe = PipelineBuilder.build_pipeline(X, "random_forest", "classification")

        assert pipe is not None
        assert len(pipe.steps) == 2  # preprocessor + model

    def test_builds_regression_pipeline(self, regression_df):
        """Should build a working regression pipeline."""
        X = regression_df.drop(columns=["target"])
        pipe = PipelineBuilder.build_pipeline(X, "linear_regression", "regression")

        assert pipe is not None

    def test_pipeline_fits_on_training_only(self, clean_classification_df):
        """Pipeline should fit on training data only."""
        X = clean_classification_df.drop(columns=["target"])
        y = clean_classification_df["target"]

        X_train, X_test, y_train, y_test = SplitManager.train_test_split(X, y)

        pipe = PipelineBuilder.build_pipeline(X_train, "logistic_regression", "classification")
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        assert len(preds) == len(X_test)
        assert all(p in [0, 1] for p in preds)


class TestSplitManager:

    def test_no_overlap_in_split(self, clean_classification_df):
        """Train and test should not overlap after splitting."""
        X = clean_classification_df.drop(columns=["target"])
        y = clean_classification_df["target"]

        X_train, X_test, _, _ = SplitManager.train_test_split(X, y)
        overlap = SplitManager.check_split_overlap(X_train, X_test)

        assert overlap["overlap_count"] == 0, "Clean split should have zero overlap"

    def test_split_preserves_data(self, clean_classification_df):
        """Split should not lose any rows."""
        X = clean_classification_df.drop(columns=["target"])
        y = clean_classification_df["target"]

        X_train, X_test, y_train, y_test = SplitManager.train_test_split(X, y)

        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)


class TestMetrics:

    def test_classification_metrics(self):
        """Should compute real classification metrics."""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])

        metrics = MetricsCalculator.classification_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "confusion_matrix" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_regression_metrics(self):
        """Should compute real regression metrics."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.1])

        metrics = MetricsCalculator.regression_metrics(y_true, y_pred)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0

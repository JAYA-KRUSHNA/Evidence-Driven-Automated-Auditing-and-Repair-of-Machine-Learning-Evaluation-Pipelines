"""
Phases 8–12 — Tests for ML modules:
    TaskIdentifier, ClassificationEngine, RegressionEngine,
    ComparisonEngine, PredictionEngine
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

_proj = Path(__file__).resolve().parents[2]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

from universal_ml_studio.ml.task_identifier import TaskIdentifier
from universal_ml_studio.ml.classification import ClassificationEngine
from universal_ml_studio.ml.regression import RegressionEngine
from universal_ml_studio.ml.comparison import ComparisonEngine
from universal_ml_studio.ml.prediction import PredictionEngine


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _clf_df(rows=200):
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "f1": rng.randn(rows),
        "f2": rng.randn(rows) + 2,
        "f3": rng.randn(rows) * 3,
        "target": rng.choice([0, 1], size=rows, p=[0.5, 0.5]),
    })


def _reg_df(rows=200):
    rng = np.random.RandomState(42)
    X = rng.randn(rows, 3)
    y = X[:, 0] * 2 + X[:, 1] - X[:, 2] * 0.5 + rng.randn(rows) * 0.1
    return pd.DataFrame({
        "f1": X[:, 0], "f2": X[:, 1], "f3": X[:, 2], "target": y,
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 8 — Task Identifier
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaskIdentifier:

    def test_binary_int_is_classification(self):
        df = _clf_df()
        result = TaskIdentifier.infer(df, "target")
        assert result["task_type"] == "classification"
        assert result["confidence"] in ("high", "medium")

    def test_continuous_float_is_regression(self):
        df = _reg_df()
        result = TaskIdentifier.infer(df, "target")
        assert result["task_type"] == "regression"

    def test_string_target_is_classification(self):
        df = pd.DataFrame({
            "f1": [1, 2, 3, 4],
            "target": ["cat", "dog", "cat", "dog"],
        })
        result = TaskIdentifier.infer(df, "target")
        assert result["task_type"] == "classification"
        assert result["confidence"] == "high"

    def test_missing_column_raises(self):
        df = _clf_df()
        with pytest.raises(ValueError):
            TaskIdentifier.infer(df, "nonexistent")

    def test_result_has_required_keys(self):
        df = _clf_df()
        result = TaskIdentifier.infer(df, "target")
        for key in ("task_type", "confidence", "rationale", "nunique", "target_dtype"):
            assert key in result


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 9 — Classification Engine
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassificationEngine:

    def test_available_algorithms(self):
        algos = ClassificationEngine.available_algorithms()
        assert "random_forest" in algos
        assert "logistic_regression" in algos
        assert len(algos) >= 9

    def test_train_random_forest(self):
        df = _clf_df()
        result = ClassificationEngine.train(df, "target", "random_forest")
        assert result.error == ""
        assert "accuracy" in result.metrics
        assert result.metrics["accuracy"] > 0.5  # better than random

    def test_train_logistic_regression(self):
        df = _clf_df()
        result = ClassificationEngine.train(df, "target", "logistic_regression")
        assert result.error == ""
        assert "accuracy" in result.metrics

    def test_cv_scores_present(self):
        df = _clf_df()
        result = ClassificationEngine.train(df, "target", "decision_tree")
        assert len(result.cv_scores) > 0
        assert "cv_mean_accuracy" in result.metrics

    def test_model_object_returned(self):
        df = _clf_df()
        result = ClassificationEngine.train(df, "target", "naive_bayes")
        assert result.model is not None

    def test_feature_columns_recorded(self):
        df = _clf_df()
        result = ClassificationEngine.train(df, "target", "knn")
        assert "f1" in result.feature_columns
        assert "target" not in result.feature_columns

    def test_unknown_algorithm_raises(self):
        df = _clf_df()
        result = ClassificationEngine.train(df, "target", "nonexistent_algo")
        assert result.error != ""

    def test_training_time_positive(self):
        df = _clf_df()
        result = ClassificationEngine.train(df, "target", "gradient_boosting")
        assert result.training_time_sec >= 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 10 — Regression Engine
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegressionEngine:

    def test_available_algorithms(self):
        algos = RegressionEngine.available_algorithms()
        assert "random_forest" in algos
        assert "linear_regression" in algos
        assert len(algos) >= 10

    def test_train_linear_regression(self):
        df = _reg_df()
        result = RegressionEngine.train(df, "target", "linear_regression")
        assert result.error == ""
        assert "r2" in result.metrics
        assert result.metrics["r2"] > 0.9  # near-linear data should fit well

    def test_train_random_forest(self):
        df = _reg_df()
        result = RegressionEngine.train(df, "target", "random_forest")
        assert result.error == ""
        assert "rmse" in result.metrics
        assert "mae" in result.metrics

    def test_cv_scores_present(self):
        df = _reg_df()
        result = RegressionEngine.train(df, "target", "ridge")
        assert len(result.cv_scores) > 0
        assert "cv_mean_rmse" in result.metrics

    def test_model_object_returned(self):
        df = _reg_df()
        result = RegressionEngine.train(df, "target", "lasso")
        assert result.model is not None

    def test_training_time_positive(self):
        df = _reg_df()
        result = RegressionEngine.train(df, "target", "knn")
        assert result.training_time_sec >= 0

    def test_mse_rmse_consistency(self):
        df = _reg_df()
        result = RegressionEngine.train(df, "target", "linear_regression")
        mse = result.metrics.get("mse", 0)
        rmse = result.metrics.get("rmse", 0)
        assert abs(rmse - mse ** 0.5) < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 11 — Comparison Engine
# ═══════════════════════════════════════════════════════════════════════════════

class TestComparisonEngine:

    def test_classification_comparison(self):
        df = _clf_df()
        result = ComparisonEngine.compare(
            df, "target", task_type="classification",
            algorithms=["logistic_regression", "decision_tree", "random_forest"],
        )
        assert result.best_algorithm != ""
        assert len(result.results) == 3

    def test_regression_comparison(self):
        df = _reg_df()
        result = ComparisonEngine.compare(
            df, "target", task_type="regression",
            algorithms=["linear_regression", "ridge", "random_forest"],
        )
        assert result.best_algorithm != ""
        assert result.ranking_metric == "r2"

    def test_results_ordered_by_metric(self):
        df = _clf_df()
        result = ComparisonEngine.compare(
            df, "target", task_type="classification",
            algorithms=["logistic_regression", "random_forest"],
        )
        if len(result.results) >= 2:
            acc0 = result.results[0].get("metrics", {}).get("accuracy", 0)
            acc1 = result.results[1].get("metrics", {}).get("accuracy", 0)
            assert acc0 >= acc1

    def test_best_metrics_populated(self):
        df = _clf_df()
        result = ComparisonEngine.compare(
            df, "target", task_type="classification",
            algorithms=["random_forest"],
        )
        assert "accuracy" in result.best_metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 12 — Prediction Engine
# ═══════════════════════════════════════════════════════════════════════════════

class TestPredictionEngine:

    def _get_trained_model(self):
        df = _clf_df()
        tr = ClassificationEngine.train(df, "target", "random_forest",
                                        config={"cv_folds": 2})
        return tr.model, tr.feature_columns, tr.label_encoder

    def test_single_prediction_classification(self):
        model, features, le = self._get_trained_model()
        sample = {col: 0.5 for col in features}
        result = PredictionEngine.predict(
            model, sample, features, label_encoder=le, task_type="classification",
        )
        assert "prediction" in result
        assert "probabilities" in result
        assert "confidence" in result

    def test_prediction_is_valid_class(self):
        model, features, le = self._get_trained_model()
        sample = {col: 0.0 for col in features}
        result = PredictionEngine.predict(
            model, sample, features, label_encoder=le, task_type="classification",
        )
        pred = str(result["prediction"])
        assert pred in ("0", "1")

    def test_regression_prediction(self):
        df = _reg_df()
        tr = RegressionEngine.train(df, "target", "linear_regression",
                                    config={"cv_folds": 2})
        sample = {"f1": 0.5, "f2": -0.3, "f3": 1.2}
        result = PredictionEngine.predict(
            tr.model, sample, tr.feature_columns, task_type="regression",
        )
        assert "prediction" in result
        assert isinstance(result["prediction"], float)

    def test_missing_features_handled(self):
        """Missing features should default to 0 without crashing."""
        model, features, le = self._get_trained_model()
        # Provide only partial features
        sample = {features[0]: 1.0}
        result = PredictionEngine.predict(
            model, sample, features, label_encoder=le, task_type="classification",
        )
        assert "prediction" in result

    def test_batch_prediction(self):
        model, features, le = self._get_trained_model()
        df_input = pd.DataFrame([{col: 0.0 for col in features}] * 5)
        results = PredictionEngine.batch_predict(
            model, df_input, features, label_encoder=le, task_type="classification",
        )
        assert len(results) == 5
        for r in results:
            assert "prediction" in r

"""
Trainer — Model training with proper pipeline handling.

Trains scikit-learn models within Pipelines to ensure:
- Preprocessing is fitted only on training data
- Cross-validation is handled correctly
- All metrics are from real predictions
"""

import logging
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

from src.experiments.pipeline_builder import PipelineBuilder
from src.experiments.metrics import MetricsCalculator
from src.config import DEFAULT_RANDOM_SEED, DEFAULT_CV_FOLDS

logger = logging.getLogger(__name__)


class TrainingResult:
    """Container for training results."""

    def __init__(self):
        self.model_name: str = ""
        self.task_type: str = ""
        self.pipeline = None
        self.train_metrics: dict = {}
        self.test_metrics: dict = {}
        self.cv_metrics: dict = {}
        self.cv_fold_scores: list = []
        self.baseline_metrics: dict = {}
        self.training_time: float = 0.0
        self.feature_importances: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "task_type": self.task_type,
            "train_metrics": self.train_metrics,
            "test_metrics": self.test_metrics,
            "cv_metrics": self.cv_metrics,
            "baseline_metrics": self.baseline_metrics,
            "training_time": self.training_time,
        }


class Trainer:
    """
    Trains ML models using sklearn Pipelines.
    All training uses real data and real model fitting.
    """

    def __init__(self, random_seed: int = DEFAULT_RANDOM_SEED):
        self.random_seed = random_seed

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        model_name: str,
        task_type: str = "classification",
        n_cv_folds: int = DEFAULT_CV_FOLDS,
    ) -> TrainingResult:
        """
        Train a model and evaluate on train, test, and cross-validation.

        All metrics are computed from real predictions.
        """
        result = TrainingResult()
        result.model_name = model_name
        result.task_type = task_type

        start_time = time.time()

        # Build pipeline
        pipeline = PipelineBuilder.build_pipeline(
            X_train, model_name, task_type, self.random_seed
        )

        # Fit on training data
        logger.info(f"Training {model_name} on {len(X_train)} samples...")
        pipeline.fit(X_train, y_train)
        result.pipeline = pipeline

        # Training predictions & metrics
        y_train_pred = pipeline.predict(X_train)
        y_train_prob = self._get_probabilities(pipeline, X_train, task_type)
        result.train_metrics = MetricsCalculator.compute_metrics(
            y_train.values, y_train_pred, y_train_prob, task_type
        )

        # Test predictions & metrics
        y_test_pred = pipeline.predict(X_test)
        y_test_prob = self._get_probabilities(pipeline, X_test, task_type)
        result.test_metrics = MetricsCalculator.compute_metrics(
            y_test.values, y_test_pred, y_test_prob, task_type
        )

        # Cross-validation on training data
        result.cv_metrics, result.cv_fold_scores = self._cross_validate(
            X_train, y_train, model_name, task_type, n_cv_folds
        )

        # Baseline comparison
        result.baseline_metrics = self._train_baseline(
            X_train, X_test, y_train, y_test, task_type
        )

        # Feature importances
        result.feature_importances = self._get_feature_importances(
            pipeline, X_train
        )

        result.training_time = round(time.time() - start_time, 2)
        logger.info(
            f"Training complete in {result.training_time}s. "
            f"Test {self._primary_metric(task_type)}: "
            f"{result.test_metrics.get(self._primary_metric(task_type), 'N/A')}"
        )

        return result

    def _get_probabilities(self, pipeline, X, task_type):
        """Safely get prediction probabilities."""
        if task_type != "classification":
            return None
        try:
            if hasattr(pipeline, "predict_proba"):
                return pipeline.predict_proba(X)
            elif hasattr(pipeline, "decision_function"):
                return pipeline.decision_function(X)
        except Exception:
            pass
        return None

    def _cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        task_type: str,
        n_folds: int,
    ) -> Tuple[dict, list]:
        """Perform cross-validation and return aggregate metrics."""
        pipeline = PipelineBuilder.build_pipeline(
            X, model_name, task_type, self.random_seed
        )

        if task_type == "classification":
            scoring = {
                "accuracy": "accuracy",
                "f1": "f1_weighted",
                "precision": "precision_weighted",
                "recall": "recall_weighted",
            }
            # Add AUC for binary
            if y.nunique() == 2:
                scoring["roc_auc"] = "roc_auc"
        else:
            scoring = {
                "r2": "r2",
                "neg_mae": "neg_mean_absolute_error",
                "neg_rmse": "neg_root_mean_squared_error",
            }

        try:
            from sklearn.model_selection import StratifiedKFold, KFold

            if task_type == "classification":
                cv = StratifiedKFold(
                    n_splits=min(n_folds, len(y)),
                    shuffle=True,
                    random_state=self.random_seed
                )
            else:
                cv = KFold(
                    n_splits=min(n_folds, len(y)),
                    shuffle=True,
                    random_state=self.random_seed
                )

            cv_results = cross_validate(
                pipeline, X, y, cv=cv, scoring=scoring,
                return_train_score=False, n_jobs=-1,
                error_score="raise",
            )

            # Aggregate results
            cv_metrics = {}
            fold_scores = []

            for metric_key, values in cv_results.items():
                if metric_key.startswith("test_"):
                    clean_name = metric_key.replace("test_", "")
                    # Convert negative metrics
                    if "neg_" in clean_name:
                        values = -values
                        clean_name = clean_name.replace("neg_", "")
                    cv_metrics[f"cv_{clean_name}_mean"] = float(np.mean(values))
                    cv_metrics[f"cv_{clean_name}_std"] = float(np.std(values))

                    for i, v in enumerate(values):
                        fold_scores.append({
                            "fold": i + 1,
                            "metric": clean_name,
                            "value": float(v) if "neg_" not in metric_key else float(-v) if v < 0 else float(v),
                        })

            return cv_metrics, fold_scores

        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return {}, []

    def _train_baseline(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        task_type: str,
    ) -> dict:
        """Train and evaluate a dummy baseline model."""
        try:
            baseline = PipelineBuilder.build_baseline_pipeline(
                X_train, task_type, self.random_seed
            )
            baseline.fit(X_train, y_train)
            y_pred = baseline.predict(X_test)
            y_prob = self._get_probabilities(baseline, X_test, task_type)

            return MetricsCalculator.compute_metrics(
                y_test.values, y_pred, y_prob, task_type
            )
        except Exception as e:
            logger.warning(f"Baseline training failed: {e}")
            return {}

    def _get_feature_importances(
        self, pipeline, X: pd.DataFrame
    ) -> Optional[dict]:
        """Extract feature importances if available."""
        try:
            model = pipeline.named_steps.get("model")
            if model is None:
                return None

            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_

                # Get feature names from preprocessor
                preprocessor = pipeline.named_steps.get("preprocessor")
                if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
                    try:
                        feature_names = list(preprocessor.get_feature_names_out())
                    except Exception:
                        feature_names = [f"feature_{i}" for i in range(len(importances))]
                else:
                    feature_names = [f"feature_{i}" for i in range(len(importances))]

                # Sort by importance
                if len(feature_names) == len(importances):
                    importance_dict = dict(zip(feature_names, importances.tolist()))
                    sorted_imp = dict(
                        sorted(importance_dict.items(),
                               key=lambda x: abs(x[1]), reverse=True)[:20]
                    )
                    return sorted_imp

            elif hasattr(model, "coef_"):
                coefs = model.coef_
                if coefs.ndim > 1:
                    coefs = np.mean(np.abs(coefs), axis=0)
                else:
                    coefs = np.abs(coefs)

                preprocessor = pipeline.named_steps.get("preprocessor")
                if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
                    try:
                        feature_names = list(preprocessor.get_feature_names_out())
                    except Exception:
                        feature_names = [f"feature_{i}" for i in range(len(coefs))]
                else:
                    feature_names = [f"feature_{i}" for i in range(len(coefs))]

                if len(feature_names) == len(coefs):
                    importance_dict = dict(zip(feature_names, coefs.tolist()))
                    sorted_imp = dict(
                        sorted(importance_dict.items(),
                               key=lambda x: abs(x[1]), reverse=True)[:20]
                    )
                    return sorted_imp

        except Exception as e:
            logger.debug(f"Could not extract feature importances: {e}")

        return None

    @staticmethod
    def _primary_metric(task_type: str) -> str:
        return "f1_score" if task_type == "classification" else "r2"

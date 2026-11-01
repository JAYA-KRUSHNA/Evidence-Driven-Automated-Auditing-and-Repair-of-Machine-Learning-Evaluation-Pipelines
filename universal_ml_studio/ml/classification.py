"""
Classification Engine — 9 sklearn classifiers with cross-validation.

Algorithms
----------
    logistic_regression, decision_tree, random_forest, knn, svm,
    naive_bayes, gradient_boosting, extra_trees, sgd

Public API
----------
    ClassificationEngine.train(df, target_column, algorithm, config)
        → TrainingResult (dataclass)
    ClassificationEngine.available_algorithms() → list[str]
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import cross_val_predict
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ..config import (
    CLASSIFICATION_ALGORITHMS,
    DEFAULT_TEST_SIZE,
    DEFAULT_CV_FOLDS,
    DEFAULT_RANDOM_SEED,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    algorithm: str
    task_type: str = "classification"
    metrics: dict = field(default_factory=dict)
    cv_scores: list[float] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    target_column: str = ""
    training_time_sec: float = 0.0
    model: Any = None
    label_encoder: Any = None
    error: str = ""


class ClassificationEngine:
    """Train and evaluate classification models."""

    @staticmethod
    def available_algorithms() -> list[str]:
        return list(CLASSIFICATION_ALGORITHMS)

    @staticmethod
    def train(
        df: pd.DataFrame,
        target_column: str,
        algorithm: str = "random_forest",
        config: dict | None = None,
    ) -> TrainingResult:
        """
        Train a classifier and return metrics + model object.

        config keys:
            test_size, cv_folds, random_seed, hyperparameters (dict)
        """
        config = config or {}
        test_size = config.get("test_size", DEFAULT_TEST_SIZE)
        cv_folds  = config.get("cv_folds", DEFAULT_CV_FOLDS)
        seed      = config.get("random_seed", DEFAULT_RANDOM_SEED)
        hyperparams = config.get("hyperparameters", {})

        result = TrainingResult(algorithm=algorithm, target_column=target_column)

        try:
            # Prepare data
            X, y, feature_cols, le = ClassificationEngine._prepare(df, target_column)
            result.feature_columns = feature_cols
            result.label_encoder = le

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed, stratify=y if len(np.unique(y)) > 1 else None
            )

            # Build model
            model = ClassificationEngine._build_model(algorithm, seed, hyperparams)
            result.model = model

            # Train
            t0 = time.time()
            model.fit(X_train, y_train)
            result.training_time_sec = round(time.time() - t0, 4)

            # Predict
            y_pred = model.predict(X_test)
            multiclass = len(np.unique(y)) > 2
            avg = "weighted" if multiclass else "binary"

            metrics: dict[str, Any] = {
                "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
                "precision": round(float(precision_score(y_test, y_pred, average=avg, zero_division=0)), 4),
                "recall":    round(float(recall_score(y_test, y_pred, average=avg, zero_division=0)), 4),
                "f1":        round(float(f1_score(y_test, y_pred, average=avg, zero_division=0)), 4),
            }

            # ROC-AUC (binary only)
            try:
                if not multiclass and hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_test)[:, 1]
                    metrics["roc_auc"] = round(float(roc_auc_score(y_test, proba)), 4)
            except Exception:
                pass

            # Confusion matrix
            try:
                metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
            except Exception:
                pass

            result.metrics = metrics

            # Cross-validation
            try:
                cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="accuracy")
                result.cv_scores = [round(float(s), 4) for s in cv_scores]
                metrics["cv_mean_accuracy"] = round(float(cv_scores.mean()), 4)
                metrics["cv_std_accuracy"]  = round(float(cv_scores.std()), 4)
            except Exception:
                pass

            logger.info(
                "Trained %s — accuracy=%.4f, time=%.2fs",
                algorithm, metrics.get("accuracy", 0), result.training_time_sec,
            )

        except Exception as exc:
            result.error = str(exc)
            logger.error("Training failed for %s: %s", algorithm, exc)

        return result

    # ── Private ───────────────────────────────────────────────────────────

    @staticmethod
    def _prepare(df: pd.DataFrame, target_column: str):
        """Drop non-numeric columns, encode target, return (X, y, cols, le)."""
        data = df.copy()
        # Drop rows where target is missing
        data = data.dropna(subset=[target_column])

        y_raw = data[target_column]
        X_df = data.drop(columns=[target_column])

        # Keep only numeric columns for X
        X_df = X_df.select_dtypes(include=["number"]).fillna(0)
        feature_cols = X_df.columns.tolist()

        X = X_df.values

        # Encode target
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))

        return X, y, feature_cols, le

    @staticmethod
    def _build_model(algorithm: str, seed: int, hyperparams: dict):
        factories = {
            "logistic_regression": lambda: LogisticRegression(
                random_state=seed, max_iter=1000,
                **{k: v for k, v in hyperparams.items() if k in ("C", "solver", "penalty")}
            ),
            "decision_tree": lambda: DecisionTreeClassifier(
                random_state=seed,
                **{k: v for k, v in hyperparams.items() if k in ("max_depth", "min_samples_split")}
            ),
            "random_forest": lambda: RandomForestClassifier(
                n_estimators=hyperparams.get("n_estimators", 100), random_state=seed,
                **{k: v for k, v in hyperparams.items() if k in ("max_depth", "min_samples_split")}
            ),
            "knn": lambda: KNeighborsClassifier(
                n_neighbors=hyperparams.get("n_neighbors", 5),
            ),
            "svm": lambda: SVC(
                probability=True, random_state=seed,
                **{k: v for k, v in hyperparams.items() if k in ("C", "kernel", "gamma")}
            ),
            "naive_bayes": lambda: GaussianNB(),
            "gradient_boosting": lambda: GradientBoostingClassifier(
                n_estimators=hyperparams.get("n_estimators", 100), random_state=seed,
            ),
            "extra_trees": lambda: ExtraTreesClassifier(
                n_estimators=hyperparams.get("n_estimators", 100), random_state=seed,
            ),
            "sgd": lambda: SGDClassifier(
                random_state=seed, max_iter=1000,
            ),
        }
        if algorithm not in factories:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Available: {list(factories.keys())}"
            )
        return factories[algorithm]()

"""
Regression Engine — 10 sklearn regressors with cross-validation.

Algorithms
----------
    linear_regression, ridge, lasso, decision_tree, random_forest,
    knn, svr, gradient_boosting, extra_trees, sgd

Public API
----------
    RegressionEngine.train(df, target_column, algorithm, config)
        → RegressionResult (dataclass)
    RegressionEngine.available_algorithms() → list[str]
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from ..config import (
    REGRESSION_ALGORITHMS,
    DEFAULT_TEST_SIZE,
    DEFAULT_CV_FOLDS,
    DEFAULT_RANDOM_SEED,
)

logger = logging.getLogger(__name__)


@dataclass
class RegressionResult:
    algorithm: str
    task_type: str = "regression"
    metrics: dict = field(default_factory=dict)
    cv_scores: list[float] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    target_column: str = ""
    training_time_sec: float = 0.0
    model: Any = None
    error: str = ""


class RegressionEngine:
    """Train and evaluate regression models."""

    @staticmethod
    def available_algorithms() -> list[str]:
        return list(REGRESSION_ALGORITHMS)

    @staticmethod
    def train(
        df: pd.DataFrame,
        target_column: str,
        algorithm: str = "random_forest",
        config: dict | None = None,
    ) -> RegressionResult:
        """
        Train a regressor and return metrics + model object.

        config keys: test_size, cv_folds, random_seed, hyperparameters (dict)
        """
        config = config or {}
        test_size   = config.get("test_size", DEFAULT_TEST_SIZE)
        cv_folds    = config.get("cv_folds", DEFAULT_CV_FOLDS)
        seed        = config.get("random_seed", DEFAULT_RANDOM_SEED)
        hyperparams = config.get("hyperparameters", {})

        result = RegressionResult(algorithm=algorithm, target_column=target_column)

        try:
            X, y, feature_cols = RegressionEngine._prepare(df, target_column)
            result.feature_columns = feature_cols

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed
            )

            model = RegressionEngine._build_model(algorithm, seed, hyperparams)
            result.model = model

            t0 = time.time()
            model.fit(X_train, y_train)
            result.training_time_sec = round(time.time() - t0, 4)

            y_pred = model.predict(X_test)

            mae  = float(mean_absolute_error(y_test, y_pred))
            mse  = float(mean_squared_error(y_test, y_pred))
            rmse = float(np.sqrt(mse))
            r2   = float(r2_score(y_test, y_pred))

            metrics: dict[str, Any] = {
                "mae":  round(mae, 4),
                "mse":  round(mse, 4),
                "rmse": round(rmse, 4),
                "r2":   round(r2, 4),
            }

            # MAPE (careful with zeros)
            try:
                mape = float(mean_absolute_percentage_error(y_test, y_pred))
                metrics["mape"] = round(mape, 4)
            except Exception:
                pass

            result.metrics = metrics

            # Cross-validation (neg MSE → RMSE)
            try:
                cv_scores = cross_val_score(
                    model, X, y, cv=cv_folds, scoring="neg_mean_squared_error"
                )
                cv_rmse = np.sqrt(-cv_scores)
                result.cv_scores = [round(float(s), 4) for s in cv_rmse]
                metrics["cv_mean_rmse"] = round(float(cv_rmse.mean()), 4)
                metrics["cv_std_rmse"]  = round(float(cv_rmse.std()), 4)
            except Exception:
                pass

            logger.info(
                "Trained %s — r2=%.4f, rmse=%.4f, time=%.2fs",
                algorithm, r2, rmse, result.training_time_sec,
            )

        except Exception as exc:
            result.error = str(exc)
            logger.error("Training failed for %s: %s", algorithm, exc)

        return result

    # ── Private ───────────────────────────────────────────────────────────

    @staticmethod
    def _prepare(df: pd.DataFrame, target_column: str):
        data = df.dropna(subset=[target_column]).copy()
        y = data[target_column].astype(float).values
        X_df = data.drop(columns=[target_column]).select_dtypes(include=["number"]).fillna(0)
        return X_df.values, y, X_df.columns.tolist()

    @staticmethod
    def _build_model(algorithm: str, seed: int, hyperparams: dict):
        factories = {
            "linear_regression": lambda: LinearRegression(),
            "ridge": lambda: Ridge(
                alpha=hyperparams.get("alpha", 1.0),
            ),
            "lasso": lambda: Lasso(
                alpha=hyperparams.get("alpha", 1.0), max_iter=2000,
            ),
            "decision_tree": lambda: DecisionTreeRegressor(
                random_state=seed,
                **{k: v for k, v in hyperparams.items() if k in ("max_depth", "min_samples_split")}
            ),
            "random_forest": lambda: RandomForestRegressor(
                n_estimators=hyperparams.get("n_estimators", 100), random_state=seed,
            ),
            "knn": lambda: KNeighborsRegressor(
                n_neighbors=hyperparams.get("n_neighbors", 5),
            ),
            "svr": lambda: SVR(
                **{k: v for k, v in hyperparams.items() if k in ("C", "kernel", "gamma", "epsilon")}
            ),
            "gradient_boosting": lambda: GradientBoostingRegressor(
                n_estimators=hyperparams.get("n_estimators", 100), random_state=seed,
            ),
            "extra_trees": lambda: ExtraTreesRegressor(
                n_estimators=hyperparams.get("n_estimators", 100), random_state=seed,
            ),
            "sgd": lambda: SGDRegressor(
                random_state=seed, max_iter=1000,
            ),
        }
        if algorithm not in factories:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Available: {list(factories.keys())}"
            )
        return factories[algorithm]()

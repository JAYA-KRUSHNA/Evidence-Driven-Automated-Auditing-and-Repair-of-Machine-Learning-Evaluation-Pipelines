"""
Comparison Engine — train multiple algorithms and rank them.

Public API
----------
    ComparisonEngine.compare(df, target_column, task_type, algorithms, config)
        → ComparisonResult (dataclass)
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .classification import ClassificationEngine
from .regression import RegressionEngine

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    task_type: str
    target_column: str
    algorithms_run: list[str] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    best_algorithm: str = ""
    best_metrics: dict = field(default_factory=dict)
    ranking_metric: str = ""
    errors: list[str] = field(default_factory=list)


class ComparisonEngine:
    """Run multiple ML algorithms and rank them by a chosen metric."""

    @staticmethod
    def compare(
        df: pd.DataFrame,
        target_column: str,
        task_type: str = "classification",
        algorithms: list[str] | None = None,
        config: dict | None = None,
    ) -> ComparisonResult:
        """
        Train all specified algorithms and produce a ranked comparison.

        Parameters
        ----------
        df           : preprocessed DataFrame
        target_column: target column name
        task_type    : 'classification' | 'regression'
        algorithms   : list of algorithm names; None → use defaults
        config       : training config (test_size, cv_folds, random_seed)
        """
        config = config or {}
        result = ComparisonResult(task_type=task_type, target_column=target_column)

        # Choose default algorithms
        if algorithms is None:
            if task_type == "classification":
                algorithms = ["logistic_regression", "decision_tree", "random_forest",
                              "knn", "naive_bayes", "gradient_boosting"]
            else:
                algorithms = ["linear_regression", "ridge", "decision_tree",
                              "random_forest", "knn", "gradient_boosting"]

        result.algorithms_run = algorithms
        result.ranking_metric = "accuracy" if task_type == "classification" else "r2"

        for algo in algorithms:
            logger.info("Comparing: %s", algo)
            try:
                if task_type == "classification":
                    tr = ClassificationEngine.train(df, target_column, algo, config)
                    row = {
                        "algorithm": algo,
                        "metrics": tr.metrics,
                        "training_time_sec": tr.training_time_sec,
                        "cv_scores": tr.cv_scores,
                        "error": tr.error,
                    }
                else:
                    tr = RegressionEngine.train(df, target_column, algo, config)
                    row = {
                        "algorithm": algo,
                        "metrics": tr.metrics,
                        "training_time_sec": tr.training_time_sec,
                        "cv_scores": tr.cv_scores,
                        "error": tr.error,
                    }
                result.results.append(row)
                if tr.error:
                    result.errors.append(f"{algo}: {tr.error}")
            except Exception as exc:
                result.errors.append(f"{algo}: {exc}")
                result.results.append({
                    "algorithm": algo, "metrics": {}, "training_time_sec": 0,
                    "cv_scores": [], "error": str(exc),
                })

        # Rank by ranking metric
        metric = result.ranking_metric
        valid = [r for r in result.results if metric in r.get("metrics", {})]
        if valid:
            # Higher is better for accuracy/r2; lower for rmse/mae (handle both)
            reverse = metric not in ("mae", "mse", "rmse", "mape")
            valid.sort(key=lambda x: x["metrics"].get(metric, 0), reverse=reverse)
            result.results = valid + [r for r in result.results if r not in valid]
            result.best_algorithm = valid[0]["algorithm"]
            result.best_metrics = valid[0]["metrics"]

        logger.info(
            "Comparison done — best: %s (%s=%.4f)",
            result.best_algorithm, metric,
            result.best_metrics.get(metric, 0),
        )
        return result

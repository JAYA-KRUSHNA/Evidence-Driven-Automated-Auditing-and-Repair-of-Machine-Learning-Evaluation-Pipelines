"""
Metrics — Calculation and aggregation of ML evaluation metrics.

Supports both classification and regression metrics.
All values are computed from real predictions — never fabricated.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    # Classification
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report,
    # Regression
    mean_absolute_error, mean_squared_error, r2_score,
    median_absolute_error, explained_variance_score,
)

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculates comprehensive ML metrics from real predictions."""

    @staticmethod
    def classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Calculate classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional, for AUC)

        Returns:
            Dict of metric_name -> metric_value
        """
        metrics = {}
        n_classes = len(np.unique(y_true))
        is_binary = n_classes == 2

        # Basic metrics
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

        # Precision, Recall, F1
        average = "binary" if is_binary else "weighted"
        metrics["precision"] = float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        )
        metrics["recall"] = float(
            recall_score(y_true, y_pred, average=average, zero_division=0)
        )
        metrics["f1_score"] = float(
            f1_score(y_true, y_pred, average=average, zero_division=0)
        )

        # Also compute macro for multi-class
        if not is_binary:
            metrics["precision_macro"] = float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            )
            metrics["recall_macro"] = float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            )
            metrics["f1_macro"] = float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            )

        # AUC metrics (require probabilities)
        if y_prob is not None:
            try:
                if is_binary:
                    # Use the positive class probability
                    if y_prob.ndim == 2:
                        prob_positive = y_prob[:, 1]
                    else:
                        prob_positive = y_prob
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_true, prob_positive)
                    )
                    metrics["pr_auc"] = float(
                        average_precision_score(y_true, prob_positive)
                    )
                else:
                    metrics["roc_auc_ovr"] = float(
                        roc_auc_score(y_true, y_prob, multi_class="ovr",
                                     average="weighted")
                    )
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not compute AUC: {e}")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        return metrics

    @staticmethod
    def regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict:
        """
        Calculate regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dict of metric_name -> metric_value
        """
        metrics = {}

        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["mse"] = float(mean_squared_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))
        metrics["median_ae"] = float(median_absolute_error(y_true, y_pred))
        metrics["explained_variance"] = float(
            explained_variance_score(y_true, y_pred)
        )

        # Relative errors
        y_range = float(np.ptp(y_true))  # Range of true values
        if y_range > 0:
            metrics["normalized_rmse"] = metrics["rmse"] / y_range

        return metrics

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        task_type: str = "classification",
    ) -> dict:
        """
        Compute metrics based on task type.

        Returns:
            Dict of metric_name -> metric_value
        """
        if task_type == "classification":
            return MetricsCalculator.classification_metrics(
                y_true, y_pred, y_prob
            )
        else:
            return MetricsCalculator.regression_metrics(y_true, y_pred)

    @staticmethod
    def format_metrics_table(metrics: dict, task_type: str = "classification") -> pd.DataFrame:
        """Format metrics as a readable DataFrame."""
        # Exclude confusion matrix from table display
        display_metrics = {
            k: v for k, v in metrics.items()
            if k != "confusion_matrix" and isinstance(v, (int, float))
        }

        df = pd.DataFrame([
            {"Metric": k.replace("_", " ").title(), "Value": round(v, 4)}
            for k, v in display_metrics.items()
        ])

        return df

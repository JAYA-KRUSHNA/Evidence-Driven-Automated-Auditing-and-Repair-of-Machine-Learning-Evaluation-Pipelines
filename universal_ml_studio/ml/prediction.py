"""
Prediction Engine — run inference using a trained model.

Public API
----------
    PredictionEngine.predict(model, feature_values, feature_columns,
                             label_encoder=None, task_type='classification')
        → dict
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PredictionEngine:
    """Load a trained model and run single or batch predictions."""

    @staticmethod
    def predict(
        model: Any,
        input_data: dict[str, Any] | pd.DataFrame,
        feature_columns: list[str],
        label_encoder: Any = None,
        task_type: str = "classification",
    ) -> dict[str, Any]:
        """
        Run prediction on input data.

        Parameters
        ----------
        model          : fitted sklearn estimator
        input_data     : dict (single row) or DataFrame (batch)
        feature_columns: ordered list of feature column names
        label_encoder  : fitted LabelEncoder (classification only)
        task_type      : 'classification' | 'regression'

        Returns
        -------
        dict with keys: prediction, confidence (optional), probabilities (optional)
        """
        # Normalise input
        if isinstance(input_data, dict):
            df_input = pd.DataFrame([input_data])
        else:
            df_input = input_data.copy()

        # Align columns
        for col in feature_columns:
            if col not in df_input.columns:
                df_input[col] = 0

        X = df_input[feature_columns].fillna(0).values

        if task_type == "classification":
            y_pred_encoded = model.predict(X)

            # Decode labels
            if label_encoder is not None:
                y_pred = label_encoder.inverse_transform(y_pred_encoded)
            else:
                y_pred = y_pred_encoded

            result: dict[str, Any] = {
                "prediction": y_pred[0] if len(y_pred) == 1 else y_pred.tolist(),
                "task_type": "classification",
            }

            # Probabilities
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                result["confidence"] = round(float(proba.max(axis=1)[0]), 4)
                if label_encoder is not None:
                    classes = [str(c) for c in label_encoder.classes_]
                else:
                    classes = [str(c) for c in model.classes_]
                result["probabilities"] = {
                    cls: round(float(p), 4)
                    for cls, p in zip(classes, proba[0])
                }

        else:  # regression
            y_pred = model.predict(X)
            result = {
                "prediction": round(float(y_pred[0]), 4) if len(y_pred) == 1 else [round(float(v), 4) for v in y_pred],
                "task_type": "regression",
            }

        return result

    @staticmethod
    def batch_predict(
        model: Any,
        df: pd.DataFrame,
        feature_columns: list[str],
        label_encoder: Any = None,
        task_type: str = "classification",
    ) -> list[dict[str, Any]]:
        """Run predictions for every row of a DataFrame."""
        results = []
        for _, row in df.iterrows():
            res = PredictionEngine.predict(
                model, row.to_dict(), feature_columns, label_encoder, task_type,
            )
            results.append(res)
        return results

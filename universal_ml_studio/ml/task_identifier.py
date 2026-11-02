"""
Task Identifier — auto-detect classification vs regression.

Public API
----------
    TaskIdentifier.infer(df, target_column) → dict
"""

import pandas as pd

from ..config import MAX_UNIQUE_FOR_CLASSIFICATION


class TaskIdentifier:
    """Heuristically determine if a target column implies classification or regression."""

    @staticmethod
    def infer(df: pd.DataFrame, target_column: str) -> dict:
        """
        Analyse the target column and return:
            task_type : 'classification' | 'regression'
            confidence : 'high' | 'medium' | 'low'
            rationale  : human-readable explanation
            nunique    : number of unique target values
        """
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in DataFrame.")

        series = df[target_column].dropna()
        nunique = int(series.nunique())
        dtype = series.dtype

        # Categorical / boolean / string → classification
        if (dtype == object
                or pd.api.types.is_string_dtype(dtype)
                or str(dtype) in ("category", "bool", "boolean")):
            return {
                "task_type": "classification",
                "confidence": "high",
                "rationale": f"Target dtype is '{dtype}' — categorical.",
                "nunique": nunique,
                "target_dtype": str(dtype),
            }

        # Integer target with few unique values → classification
        if pd.api.types.is_integer_dtype(dtype) and nunique <= MAX_UNIQUE_FOR_CLASSIFICATION:
            return {
                "task_type": "classification",
                "confidence": "high",
                "rationale": f"Integer target with only {nunique} unique values.",
                "nunique": nunique,
                "target_dtype": str(dtype),
            }

        # Float target with few unique values (e.g. 0.0, 1.0) → classification
        if pd.api.types.is_float_dtype(dtype):
            int_vals = series.dropna().apply(lambda x: x == int(x))
            if int_vals.all() and nunique <= MAX_UNIQUE_FOR_CLASSIFICATION:
                return {
                    "task_type": "classification",
                    "confidence": "medium",
                    "rationale": f"Float target with {nunique} whole-number unique values.",
                    "nunique": nunique,
                    "target_dtype": str(dtype),
                }

        # Continuous numeric → regression
        confidence = "high" if nunique > MAX_UNIQUE_FOR_CLASSIFICATION else "medium"
        return {
            "task_type": "regression",
            "confidence": confidence,
            "rationale": f"Continuous numeric target with {nunique} unique values.",
            "nunique": nunique,
            "target_dtype": str(dtype),
        }

    identify = infer

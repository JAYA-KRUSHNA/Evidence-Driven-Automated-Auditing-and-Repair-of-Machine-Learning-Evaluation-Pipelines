"""
Split Manager — Train/test/validation splitting strategies.

Supports:
- Random split (stratified for classification)
- K-Fold cross-validation (stratified for classification)
- Time-based split (for temporal data)
- Group-based split (for grouped entities)
"""

import logging
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold,
    TimeSeriesSplit, GroupKFold, GroupShuffleSplit
)

from src.config import DEFAULT_TEST_SIZE, DEFAULT_CV_FOLDS, DEFAULT_RANDOM_SEED

logger = logging.getLogger(__name__)


class SplitManager:
    """Manages dataset splitting with configurable strategies."""

    @staticmethod
    def train_test_split(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = DEFAULT_TEST_SIZE,
        random_seed: int = DEFAULT_RANDOM_SEED,
        stratify: bool = True,
        task_type: str = "classification"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Standard train/test split.
        Stratified by default for classification tasks.
        """
        stratify_col = y if (stratify and task_type == "classification") else None

        # Check if stratification is possible (need at least 2 samples per class)
        if stratify_col is not None:
            min_class_count = y.value_counts().min()
            if min_class_count < 2:
                logger.warning(
                    "Some classes have fewer than 2 samples. "
                    "Disabling stratification."
                )
                stratify_col = None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_seed,
            stratify=stratify_col,
        )

        logger.info(
            f"Split: train={len(X_train)}, test={len(X_test)}, "
            f"stratified={stratify_col is not None}"
        )

        return X_train, X_test, y_train, y_test

    @staticmethod
    def get_cv_splitter(
        task_type: str = "classification",
        n_splits: int = DEFAULT_CV_FOLDS,
        random_seed: int = DEFAULT_RANDOM_SEED,
        strategy: str = "auto",
        groups: Optional[pd.Series] = None,
    ):
        """
        Get a cross-validation splitter based on task type and strategy.

        Args:
            task_type: "classification" or "regression"
            n_splits: Number of CV folds
            random_seed: Random state
            strategy: "auto", "stratified", "kfold", "timeseries", "group"
            groups: Group labels for group-based splitting

        Returns:
            CV splitter object
        """
        if strategy == "timeseries":
            return TimeSeriesSplit(n_splits=n_splits)
        elif strategy == "group" and groups is not None:
            return GroupKFold(n_splits=min(n_splits, groups.nunique()))
        elif strategy == "stratified" or (strategy == "auto" and task_type == "classification"):
            return StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_seed
            )
        else:
            return KFold(
                n_splits=n_splits, shuffle=True, random_state=random_seed
            )

    @staticmethod
    def check_split_overlap(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> dict:
        """
        Check for overlapping rows between train and test sets.

        Returns a dict with overlap statistics.
        """
        # Hash each row for comparison
        train_hashes = set(
            pd.util.hash_pandas_object(X_train, index=False).values
        )
        test_hashes = set(
            pd.util.hash_pandas_object(X_test, index=False).values
        )

        overlap = train_hashes & test_hashes
        overlap_count = len(overlap)

        # Find the actual overlapping indices
        overlap_train_indices = []
        overlap_test_indices = []

        if overlap_count > 0:
            all_hashes_train = pd.util.hash_pandas_object(X_train, index=False)
            all_hashes_test = pd.util.hash_pandas_object(X_test, index=False)

            overlap_train_indices = list(
                all_hashes_train[all_hashes_train.isin(overlap)].index[:20]
            )
            overlap_test_indices = list(
                all_hashes_test[all_hashes_test.isin(overlap)].index[:20]
            )

        return {
            "overlap_count": overlap_count,
            "overlap_pct_of_test": round(
                overlap_count / max(len(X_test), 1) * 100, 2
            ),
            "overlap_pct_of_train": round(
                overlap_count / max(len(X_train), 1) * 100, 2
            ),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "sample_train_indices": overlap_train_indices,
            "sample_test_indices": overlap_test_indices,
        }

    @staticmethod
    def detect_temporal_columns(df: pd.DataFrame) -> list[str]:
        """Detect columns that appear to contain temporal/date information."""
        temporal_cols = []

        for col in df.columns:
            # Already datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                temporal_cols.append(col)
                continue

            # Check column name hints
            time_keywords = ["date", "time", "year", "month", "day",
                           "timestamp", "created", "updated", "period"]
            if any(kw in col.lower() for kw in time_keywords):
                # Try to parse
                try:
                    sample = df[col].dropna().head(50)
                    pd.to_datetime(sample)
                    temporal_cols.append(col)
                except (ValueError, TypeError):
                    pass

        return temporal_cols

    @staticmethod
    def detect_group_columns(df: pd.DataFrame) -> list[str]:
        """
        Detect columns that might represent groups/entities
        (e.g., patient ID, user ID).
        """
        group_candidates = []
        id_keywords = ["id", "user", "patient", "customer", "student",
                       "subject", "group", "session", "account"]

        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in id_keywords):
                series = df[col].dropna()
                unique_ratio = series.nunique() / max(len(series), 1)
                # Group columns have repeated values but aren't too unique
                if 0.01 < unique_ratio < 0.9:
                    group_candidates.append(col)

        return group_candidates

    @staticmethod
    def get_split_config(strategy: str, **kwargs) -> dict:
        """Create a serializable split configuration."""
        config = {
            "strategy": strategy,
            "test_size": kwargs.get("test_size", DEFAULT_TEST_SIZE),
            "n_splits": kwargs.get("n_splits", DEFAULT_CV_FOLDS),
            "random_seed": kwargs.get("random_seed", DEFAULT_RANDOM_SEED),
            "stratified": kwargs.get("stratified", True),
        }
        if strategy == "group":
            config["group_column"] = kwargs.get("group_column", "")
        if strategy == "timeseries":
            config["time_column"] = kwargs.get("time_column", "")
        return config

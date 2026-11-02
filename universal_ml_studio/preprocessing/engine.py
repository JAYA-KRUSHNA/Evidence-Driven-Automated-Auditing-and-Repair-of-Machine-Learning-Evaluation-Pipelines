"""
Preprocessing Engine — automated data cleaning and feature engineering.

Handles:
    - Missing value imputation (mean, median, mode, drop)
    - Categorical encoding (label, one-hot, ordinal)
    - Feature scaling (standard, min-max, robust)
    - Outlier treatment (IQR capping, z-score capping, removal)
    - Feature engineering (interaction terms, polynomial features)
    - Datetime feature extraction

All operations are recorded for reproducibility and each step
returns the transformed DataFrame plus an operation log.

Public API
----------
    PreprocessingEngine.auto_preprocess(df, metadata, target_column)
        → (DataFrame, list[operation_dict])
    PreprocessingEngine.impute_missing(df, strategy, columns)
    PreprocessingEngine.encode_categorical(df, method, columns)
    PreprocessingEngine.scale_features(df, method, columns)
    PreprocessingEngine.handle_outliers(df, method, columns)
    PreprocessingEngine.extract_datetime_features(df, columns)
    PreprocessingEngine.save_history(operations, dataset_id, session)
"""

import logging
from typing import Any

import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)

from ..config import (
    OUTLIER_IQR_MULTIPLIER,
    OUTLIER_ZSCORE_THRESHOLD,
    MISSING_VALUE_ERROR_PCT,
)

logger = logging.getLogger(__name__)


class PreprocessingEngine:
    """Stateless preprocessing utilities with operation logging."""

    # ═══════════════════════════════════════════════════════════════════════
    #  Auto-preprocess  (one-call pipeline)
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def auto_preprocess(
        df: pd.DataFrame,
        metadata: dict[str, Any],
        target_column: str | None = None,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        """
        Apply a sensible default preprocessing pipeline:
        1. Drop columns that are >80% missing
        2. Impute remaining missing (mean for numeric, mode for categorical)
        3. Encode categoricals (one-hot for low cardinality, label for high)
        4. Extract datetime features
        5. Scale numerical features (standard scaler)
        6. Cap outliers (IQR method)

        Returns (processed_df, operations_log).
        """
        result_df = df.copy()
        operations: list[dict[str, Any]] = []

        numerical_cols = metadata.get("numerical_columns", [])
        categorical_cols = metadata.get("categorical_columns", [])
        datetime_cols = metadata.get("datetime_columns", [])
        identifier_cols = metadata.get("identifier_columns", [])

        # 0. Drop identifier columns (they don't help ML)
        id_cols_to_drop = [c for c in identifier_cols if c in result_df.columns and c != target_column]
        if id_cols_to_drop:
            result_df = result_df.drop(columns=id_cols_to_drop)
            operations.append({
                "operation": "drop_identifiers",
                "category": "feature_selection",
                "parameters": {"columns": id_cols_to_drop},
                "affected_columns": id_cols_to_drop,
                "rows_before": len(result_df),
                "rows_after": len(result_df),
            })

        # 1. Drop columns with >80% missing
        cols_to_drop = []
        for col in result_df.columns:
            missing_pct = result_df[col].isnull().mean() * 100
            if missing_pct > MISSING_VALUE_ERROR_PCT and col != target_column:
                cols_to_drop.append(col)

        if cols_to_drop:
            result_df = result_df.drop(columns=cols_to_drop)
            operations.append({
                "operation": "drop_high_missing_columns",
                "category": "missing_values",
                "parameters": {"threshold_pct": MISSING_VALUE_ERROR_PCT},
                "affected_columns": cols_to_drop,
                "rows_before": len(result_df),
                "rows_after": len(result_df),
            })

        # 2. Impute missing values
        num_cols_present = [c for c in numerical_cols if c in result_df.columns and c != target_column]
        cat_cols_present = [c for c in categorical_cols if c in result_df.columns and c != target_column]

        if num_cols_present:
            cols_with_missing = [c for c in num_cols_present if result_df[c].isnull().any()]
            if cols_with_missing:
                result_df, op = PreprocessingEngine.impute_missing(
                    result_df, strategy="median", columns=cols_with_missing,
                )
                operations.append(op)

        if cat_cols_present:
            cols_with_missing = [c for c in cat_cols_present if result_df[c].isnull().any()]
            if cols_with_missing:
                result_df, op = PreprocessingEngine.impute_missing(
                    result_df, strategy="mode", columns=cols_with_missing,
                )
                operations.append(op)

        # 3. Extract datetime features (before encoding, since these create new numeric columns)
        dt_cols_present = [c for c in datetime_cols if c in result_df.columns and c != target_column]
        if dt_cols_present:
            result_df, op = PreprocessingEngine.extract_datetime_features(
                result_df, columns=dt_cols_present,
            )
            operations.append(op)

        # 4. Encode categorical columns
        cat_cols_remaining = [
            c for c in result_df.select_dtypes(include=["object", "category"]).columns
            if c != target_column
        ]
        if cat_cols_remaining:
            low_card = [c for c in cat_cols_remaining if result_df[c].nunique(dropna=True) <= 10]
            high_card = [c for c in cat_cols_remaining if result_df[c].nunique(dropna=True) > 10]

            if low_card:
                result_df, op = PreprocessingEngine.encode_categorical(
                    result_df, method="onehot", columns=low_card,
                )
                operations.append(op)

            if high_card:
                result_df, op = PreprocessingEngine.encode_categorical(
                    result_df, method="label", columns=high_card,
                )
                operations.append(op)

        # 5. Handle outliers in numerical columns
        num_cols_final = [
            c for c in result_df.select_dtypes(include=["number"]).columns
            if c != target_column
        ]
        if num_cols_final:
            result_df, op = PreprocessingEngine.handle_outliers(
                result_df, method="iqr_cap", columns=num_cols_final,
            )
            operations.append(op)

        # 6. Scale numerical features
        num_cols_scale = [
            c for c in result_df.select_dtypes(include=["number"]).columns
            if c != target_column
        ]
        if num_cols_scale:
            result_df, op = PreprocessingEngine.scale_features(
                result_df, method="standard", columns=num_cols_scale,
            )
            operations.append(op)

        logger.info(
            "Auto-preprocessing complete: %d operations, %d→%d columns",
            len(operations), len(df.columns), len(result_df.columns),
        )
        return result_df, operations

    # ═══════════════════════════════════════════════════════════════════════
    #  Missing value imputation
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def impute_missing(
        df: pd.DataFrame,
        strategy: str = "median",
        columns: list[str] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Impute missing values.

        Strategies: mean, median, mode, zero, drop_rows, drop_cols, ffill, bfill.
        """
        result = df.copy()
        if columns is None:
            columns = [c for c in result.columns if result[c].isnull().any()]
            # For mean/median strategies restrict to numeric columns only
            if strategy in ("mean", "median"):
                columns = [c for c in columns if pd.api.types.is_numeric_dtype(result[c])]

        rows_before = len(result)
        fill_values = {}

        for col in columns:
            if col not in result.columns:
                continue

            if strategy == "mean":
                val = result[col].mean()
                result[col] = result[col].fillna(val)
                fill_values[col] = float(val) if not pd.isna(val) else 0
            elif strategy == "median":
                val = result[col].median()
                result[col] = result[col].fillna(val)
                fill_values[col] = float(val) if not pd.isna(val) else 0
            elif strategy == "mode":
                mode_val = result[col].mode()
                val = mode_val.iloc[0] if len(mode_val) > 0 else ""
                result[col] = result[col].fillna(val)
                fill_values[col] = val
            elif strategy == "zero":
                result[col] = result[col].fillna(0)
                fill_values[col] = 0
            elif strategy == "ffill":
                result[col] = result[col].ffill()
            elif strategy == "bfill":
                result[col] = result[col].bfill()

        if strategy == "drop_rows":
            result = result.dropna(subset=columns)
        elif strategy == "drop_cols":
            result = result.drop(columns=[c for c in columns if c in result.columns])

        operation = {
            "operation": f"impute_{strategy}",
            "category": "missing_values",
            "parameters": {"strategy": strategy, "fill_values": fill_values},
            "affected_columns": columns,
            "rows_before": rows_before,
            "rows_after": len(result),
        }
        return result, operation

    # ═══════════════════════════════════════════════════════════════════════
    #  Categorical encoding
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def encode_categorical(
        df: pd.DataFrame,
        method: str = "onehot",
        columns: list[str] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Encode categorical columns.

        Methods: onehot, label, ordinal, frequency.
        """
        result = df.copy()
        if columns is None:
            columns = result.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        encodings = {}

        if method == "onehot":
            for col in columns:
                if col not in result.columns:
                    continue
                dummies = pd.get_dummies(result[col], prefix=col, dtype=int)
                result = pd.concat([result.drop(columns=[col]), dummies], axis=1)
                encodings[col] = dummies.columns.tolist()

        elif method == "label":
            for col in columns:
                if col not in result.columns:
                    continue
                le = LabelEncoder()
                non_null = result[col].fillna("__MISSING__")
                result[col] = le.fit_transform(non_null.astype(str))
                encodings[col] = {
                    label: int(idx) for idx, label in enumerate(le.classes_)
                }

        elif method == "frequency":
            for col in columns:
                if col not in result.columns:
                    continue
                freq = result[col].value_counts(normalize=True)
                result[col] = result[col].map(freq).fillna(0)
                encodings[col] = freq.to_dict()

        operation = {
            "operation": f"encode_{method}",
            "category": "encoding",
            "parameters": {"method": method, "encodings": encodings},
            "affected_columns": columns,
            "rows_before": len(result),
            "rows_after": len(result),
        }
        return result, operation

    # ═══════════════════════════════════════════════════════════════════════
    #  Feature scaling
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def scale_features(
        df: pd.DataFrame,
        method: str = "standard",
        columns: list[str] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Scale numerical features.

        Methods: standard, minmax, robust.
        """
        result = df.copy()
        if columns is None:
            columns = result.select_dtypes(include=["number"]).columns.tolist()

        cols_to_scale = [c for c in columns if c in result.columns]
        if not cols_to_scale:
            return result, {
                "operation": f"scale_{method}",
                "category": "scaling",
                "parameters": {"method": method},
                "affected_columns": [],
                "rows_before": len(result),
                "rows_after": len(result),
            }

        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        result[cols_to_scale] = scaler.fit_transform(result[cols_to_scale])

        operation = {
            "operation": f"scale_{method}",
            "category": "scaling",
            "parameters": {"method": method},
            "affected_columns": cols_to_scale,
            "rows_before": len(result),
            "rows_after": len(result),
        }
        return result, operation

    # ═══════════════════════════════════════════════════════════════════════
    #  Outlier handling
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def handle_outliers(
        df: pd.DataFrame,
        method: str = "iqr_cap",
        columns: list[str] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Handle outliers in numerical columns.

        Methods: iqr_cap, zscore_cap, iqr_remove.
        """
        result = df.copy()
        if columns is None:
            columns = result.select_dtypes(include=["number"]).columns.tolist()

        rows_before = len(result)
        outlier_info = {}

        for col in columns:
            if col not in result.columns:
                continue

            series = result[col].dropna()
            if len(series) < 10:
                continue

            if method in ("iqr_cap", "iqr_remove"):
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    continue
                lower = q1 - OUTLIER_IQR_MULTIPLIER * iqr
                upper = q3 + OUTLIER_IQR_MULTIPLIER * iqr

                outlier_mask = (result[col] < lower) | (result[col] > upper)
                outlier_count = int(outlier_mask.sum())

                if outlier_count > 0:
                    if method == "iqr_cap":
                        result[col] = result[col].clip(lower, upper)
                    elif method == "iqr_remove":
                        result = result[~outlier_mask]

                    outlier_info[col] = {
                        "outlier_count": outlier_count,
                        "lower_bound": round(float(lower), 4),
                        "upper_bound": round(float(upper), 4),
                    }

            elif method == "zscore_cap":
                mean = series.mean()
                std = series.std()
                if std == 0:
                    continue
                lower = mean - OUTLIER_ZSCORE_THRESHOLD * std
                upper = mean + OUTLIER_ZSCORE_THRESHOLD * std

                outlier_mask = (result[col] < lower) | (result[col] > upper)
                outlier_count = int(outlier_mask.sum())

                if outlier_count > 0:
                    result[col] = result[col].clip(lower, upper)
                    outlier_info[col] = {
                        "outlier_count": outlier_count,
                        "lower_bound": round(float(lower), 4),
                        "upper_bound": round(float(upper), 4),
                    }

        operation = {
            "operation": f"outlier_{method}",
            "category": "outliers",
            "parameters": {"method": method, "outlier_info": outlier_info},
            "affected_columns": list(outlier_info.keys()),
            "rows_before": rows_before,
            "rows_after": len(result),
        }
        return result, operation

    # ═══════════════════════════════════════════════════════════════════════
    #  Datetime feature extraction
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def extract_datetime_features(
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Extract year, month, day, day_of_week, hour from datetime columns
        and drop the original datetime column.
        """
        result = df.copy()
        if columns is None:
            columns = result.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

        new_columns_created = []

        for col in columns:
            if col not in result.columns:
                continue

            # Try to convert to datetime if not already
            try:
                dt_series = pd.to_datetime(result[col], errors="coerce")
            except Exception:
                continue

            valid = dt_series.notna()
            if valid.sum() == 0:
                continue

            result[f"{col}_year"] = dt_series.dt.year
            result[f"{col}_month"] = dt_series.dt.month
            result[f"{col}_day"] = dt_series.dt.day
            result[f"{col}_dayofweek"] = dt_series.dt.dayofweek

            # Only extract hour if there's time info
            if dt_series.dropna().dt.hour.sum() > 0:
                result[f"{col}_hour"] = dt_series.dt.hour
                new_columns_created.extend([
                    f"{col}_year", f"{col}_month", f"{col}_day",
                    f"{col}_dayofweek", f"{col}_hour",
                ])
            else:
                new_columns_created.extend([
                    f"{col}_year", f"{col}_month", f"{col}_day",
                    f"{col}_dayofweek",
                ])

            result = result.drop(columns=[col])

        operation = {
            "operation": "extract_datetime_features",
            "category": "feature_engineering",
            "parameters": {"source_columns": columns, "new_columns": new_columns_created},
            "affected_columns": columns + new_columns_created,
            "rows_before": len(result),
            "rows_after": len(result),
        }
        return result, operation

    # ═══════════════════════════════════════════════════════════════════════
    #  Database persistence
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def save_history(
        operations: list[dict[str, Any]],
        dataset_id: str,
        session=None,
    ) -> list:
        """Persist preprocessing operations to the database."""
        from ..database.models import PreprocessingHistory
        from ..database.connection import get_session

        own_session = session is None
        if own_session:
            session = get_session()

        records = []
        try:
            for op in operations:
                record = PreprocessingHistory(
                    dataset_id=dataset_id,
                    operation=op.get("operation", "unknown"),
                    category=op.get("category", "unknown"),
                    parameters=op.get("parameters"),
                    affected_columns=op.get("affected_columns"),
                    rows_before=op.get("rows_before", 0),
                    rows_after=op.get("rows_after", 0),
                )
                session.add(record)
                records.append(record)

            if own_session:
                session.commit()

            return records

        except Exception:
            if own_session:
                session.rollback()
            raise
        finally:
            if own_session:
                session.close()

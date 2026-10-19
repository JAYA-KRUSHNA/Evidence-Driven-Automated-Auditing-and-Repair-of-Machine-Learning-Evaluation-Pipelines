"""
Correction Rules — Individual correction strategies for detected issues.

Each rule handles a specific finding category and generates a corrected
experiment configuration.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class CorrectionAction:
    """A single correction action."""

    def __init__(
        self,
        correction_type: str,
        description: str,
        changes: dict,
        finding_id: str = "",
    ):
        self.correction_type = correction_type
        self.description = description
        self.changes = changes
        self.finding_id = finding_id

    def to_dict(self) -> dict:
        return {
            "correction_type": self.correction_type,
            "description": self.description,
            "changes": self.changes,
            "finding_id": self.finding_id,
        }


class CorrectionRules:
    """
    Generates correction actions for supported findings.

    Rules preserve original data and generate modifications
    to the experiment configuration.
    """

    @staticmethod
    def remove_leaky_columns(
        df: pd.DataFrame, columns_to_remove: list[str],
        target_column: str
    ) -> tuple[pd.DataFrame, CorrectionAction]:
        """Remove confirmed leaky columns from the dataset."""
        # Safety: never remove the target
        safe_cols = [c for c in columns_to_remove if c != target_column and c in df.columns]

        if not safe_cols:
            return df, None

        df_corrected = df.drop(columns=safe_cols)

        action = CorrectionAction(
            correction_type="remove_leaky_columns",
            description=f"Removed {len(safe_cols)} leaky column(s): {', '.join(safe_cols)}",
            changes={
                "removed_columns": safe_cols,
                "remaining_columns": list(df_corrected.columns),
            },
        )

        return df_corrected, action

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, CorrectionAction]:
        """Remove duplicate rows from dataset."""
        original_len = len(df)
        df_corrected = df.drop_duplicates().reset_index(drop=True)
        removed = original_len - len(df_corrected)

        if removed == 0:
            return df, None

        action = CorrectionAction(
            correction_type="remove_duplicates",
            description=f"Removed {removed} duplicate rows ({removed/original_len*100:.1f}%).",
            changes={
                "original_rows": original_len,
                "corrected_rows": len(df_corrected),
                "removed_rows": removed,
            },
        )

        return df_corrected, action

    @staticmethod
    def remove_constant_columns(
        df: pd.DataFrame, target_column: str
    ) -> tuple[pd.DataFrame, CorrectionAction]:
        """Remove constant (single-value) columns."""
        constant_cols = [
            col for col in df.columns
            if col != target_column and df[col].nunique(dropna=True) <= 1
        ]

        if not constant_cols:
            return df, None

        df_corrected = df.drop(columns=constant_cols)

        action = CorrectionAction(
            correction_type="remove_constant_columns",
            description=f"Removed {len(constant_cols)} constant column(s).",
            changes={
                "removed_columns": constant_cols,
            },
        )

        return df_corrected, action

    @staticmethod
    def replace_infinite_values(df: pd.DataFrame) -> tuple[pd.DataFrame, CorrectionAction]:
        """Replace infinite values with NaN for imputation."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_cols = []

        df_corrected = df.copy()

        for col in numeric_cols:
            inf_mask = np.isinf(df_corrected[col])
            if inf_mask.any():
                inf_count = int(inf_mask.sum())
                inf_cols.append((col, inf_count))
                df_corrected.loc[inf_mask, col] = np.nan

        if not inf_cols:
            return df, None

        action = CorrectionAction(
            correction_type="replace_infinite_values",
            description=f"Replaced infinite values with NaN in {len(inf_cols)} column(s).",
            changes={
                "affected_columns": {col: count for col, count in inf_cols},
            },
        )

        return df_corrected, action

    @staticmethod
    def fix_split_strategy(
        current_config: dict,
        temporal_columns: list[str] = None,
        group_columns: list[str] = None,
        is_imbalanced: bool = False,
    ) -> tuple[dict, CorrectionAction]:
        """Correct the split strategy based on data characteristics."""
        corrected_config = current_config.copy()
        changes = {}

        if temporal_columns:
            corrected_config["strategy"] = "timeseries"
            corrected_config["time_column"] = temporal_columns[0]
            changes["strategy"] = "timeseries"
            changes["reason"] = f"Temporal column '{temporal_columns[0]}' detected"
        elif group_columns:
            corrected_config["strategy"] = "group"
            corrected_config["group_column"] = group_columns[0]
            changes["strategy"] = "group"
            changes["reason"] = f"Group column '{group_columns[0]}' detected"
        elif is_imbalanced:
            corrected_config["stratified"] = True
            changes["stratified"] = True
            changes["reason"] = "Class imbalance detected"
        else:
            return current_config, None

        action = CorrectionAction(
            correction_type="fix_split_strategy",
            description=f"Changed split strategy: {changes.get('reason', '')}.",
            changes=changes,
        )

        return corrected_config, action

    @staticmethod
    def remove_high_missing_columns(
        df: pd.DataFrame, threshold: float = 0.5, target_column: str = ""
    ) -> tuple[pd.DataFrame, CorrectionAction]:
        """Remove columns with very high missing rates."""
        high_missing = []
        for col in df.columns:
            if col == target_column:
                continue
            missing_pct = df[col].isna().mean()
            if missing_pct > threshold:
                high_missing.append((col, round(missing_pct * 100, 1)))

        if not high_missing:
            return df, None

        cols_to_remove = [col for col, _ in high_missing]
        df_corrected = df.drop(columns=cols_to_remove)

        action = CorrectionAction(
            correction_type="remove_high_missing_columns",
            description=f"Removed {len(cols_to_remove)} columns with >{threshold*100:.0f}% missing.",
            changes={
                "removed_columns": {col: f"{pct}% missing" for col, pct in high_missing},
                "threshold": threshold,
            },
        )

        return df_corrected, action

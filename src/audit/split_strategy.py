"""
Audit D: Split Strategy Validation

Checks whether the splitting strategy matches the data characteristics:
- Time-dependent data requiring chronological splits
- Grouped entities requiring group-aware splits
- Imbalanced data requiring stratified splits
"""

import logging

import pandas as pd
import numpy as np

from src.audit.base import BaseAudit, AuditFinding
from src.config import Severity, FindingStatus, AuditCategory
from src.experiments.split_manager import SplitManager

logger = logging.getLogger(__name__)


class SplitStrategyAudit(BaseAudit):
    """Validates that the splitting strategy is appropriate for the data."""

    CATEGORY = AuditCategory.SPLIT_STRATEGY
    FINDING_PREFIX = "SS"

    def run(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str = "classification",
        split_config: dict = None,
        **kwargs,
    ) -> list[AuditFinding]:
        findings = []

        split_config = split_config or {"strategy": "auto"}

        # Check 1: Temporal columns needing time-aware splits
        findings.extend(
            self._check_temporal_requirement(df, split_config)
        )

        # Check 2: Group columns needing group-aware splits
        findings.extend(
            self._check_group_requirement(df, split_config)
        )

        # Check 3: Stratification for imbalanced classification
        if task_type == "classification":
            findings.extend(
                self._check_stratification(df, target_column, split_config)
            )

        return findings

    def _check_temporal_requirement(
        self, df: pd.DataFrame, split_config: dict
    ) -> list[AuditFinding]:
        findings = []

        temporal_cols = SplitManager.detect_temporal_columns(df)
        current_strategy = split_config.get("strategy", "auto")

        if temporal_cols and current_strategy not in ("timeseries", "temporal"):
            findings.append(self._create_finding(
                severity=Severity.MEDIUM,
                status=FindingStatus.SUSPECTED,
                description=(
                    f"Dataset contains temporal columns ({', '.join(temporal_cols[:3])}) "
                    f"but uses '{current_strategy}' splitting."
                ),
                evidence={
                    "temporal_columns": temporal_cols,
                    "current_strategy": current_strategy,
                },
                affected_columns=temporal_cols,
                explanation=(
                    "When data has a temporal dimension, random splitting can create "
                    "temporal leakage — the model trains on future data and predicts "
                    "the past. Use chronological splitting instead."
                ),
                confidence=0.6,
                recommendation=(
                    "If the prediction task is time-dependent, use time-series "
                    "cross-validation (TimeSeriesSplit) or sort by the temporal "
                    "column before splitting."
                ),
                auto_correctable=True,
            ))

        return findings

    def _check_group_requirement(
        self, df: pd.DataFrame, split_config: dict
    ) -> list[AuditFinding]:
        findings = []

        group_cols = SplitManager.detect_group_columns(df)
        current_strategy = split_config.get("strategy", "auto")

        if group_cols and current_strategy not in ("group", "grouped"):
            # Check if any group column has repeated values
            for col in group_cols:
                series = df[col].dropna()
                n_unique = series.nunique()
                n_total = len(series)

                if n_unique < n_total * 0.5:  # Significant repetition
                    avg_per_group = n_total / max(n_unique, 1)
                    findings.append(self._create_finding(
                        severity=Severity.MEDIUM,
                        status=FindingStatus.SUSPECTED,
                        description=(
                            f"Column '{col}' appears to be a group identifier "
                            f"({n_unique} groups, avg {avg_per_group:.1f} rows/group) "
                            f"but random splitting is used."
                        ),
                        evidence={
                            "column": col,
                            "n_groups": n_unique,
                            "n_rows": n_total,
                            "avg_per_group": round(avg_per_group, 1),
                            "current_strategy": current_strategy,
                        },
                        affected_columns=[col],
                        explanation=(
                            f"The column '{col}' looks like a group identifier "
                            "(e.g., patient ID, user ID). With random splitting, "
                            "the same entity may appear in both train and test sets, "
                            "leaking entity-specific information."
                        ),
                        confidence=0.5,
                        recommendation=(
                            f"Use GroupKFold or GroupShuffleSplit with '{col}' as the "
                            "group column to ensure each entity appears in only one split."
                        ),
                        auto_correctable=True,
                    ))

        return findings

    def _check_stratification(
        self, df: pd.DataFrame, target_column: str, split_config: dict
    ) -> list[AuditFinding]:
        findings = []

        if target_column not in df.columns:
            return findings

        target = df[target_column].dropna()
        value_counts = target.value_counts(normalize=True)

        if len(value_counts) < 2:
            return findings

        min_class_pct = float(value_counts.min()) * 100
        is_stratified = split_config.get("stratified", True)

        if min_class_pct < 10 and not is_stratified:
            findings.append(self._create_finding(
                severity=Severity.MEDIUM,
                status=FindingStatus.SUSPECTED,
                description=(
                    f"Class imbalance detected (minority class: {min_class_pct:.1f}%) "
                    "without stratified splitting."
                ),
                evidence={
                    "class_distribution": {
                        str(k): round(float(v) * 100, 2)
                        for k, v in value_counts.items()
                    },
                    "minority_class_pct": round(min_class_pct, 2),
                    "stratified": is_stratified,
                },
                explanation=(
                    "When classes are imbalanced, random splitting may produce "
                    "train/test sets where the minority class is severely "
                    "underrepresented, leading to unreliable evaluation."
                ),
                confidence=0.7,
                recommendation="Enable stratified splitting to preserve class proportions.",
                auto_correctable=True,
            ))

        return findings

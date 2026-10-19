"""
Audit G: Data Quality Checks

Checks dataset quality independently of leakage:
- Missing values
- Duplicate records
- Constant features
- Unexpected data types
- Infinite values
- Invalid categorical values
- Extreme values
- Target imbalance
"""

import logging

import pandas as pd
import numpy as np

from src.audit.base import BaseAudit, AuditFinding
from src.config import (
    Severity, FindingStatus, AuditCategory,
    MISSING_VALUE_WARNING_THRESHOLD, MISSING_VALUE_CRITICAL_THRESHOLD,
    CLASS_IMBALANCE_THRESHOLD,
)

logger = logging.getLogger(__name__)


class DataQualityAudit(BaseAudit):
    """Checks data quality issues that may affect model training."""

    CATEGORY = AuditCategory.DATA_QUALITY
    FINDING_PREFIX = "DQ"

    def run(
        self,
        df: pd.DataFrame,
        target_column: str = "",
        task_type: str = "classification",
        **kwargs,
    ) -> list[AuditFinding]:
        findings = []

        # Check 1: Missing values
        findings.extend(self._check_missing_values(df))

        # Check 2: Duplicate rows
        findings.extend(self._check_duplicates(df))

        # Check 3: Constant columns
        findings.extend(self._check_constant_columns(df, target_column))

        # Check 4: Infinite values
        findings.extend(self._check_infinite_values(df))

        # Check 5: Mixed types
        findings.extend(self._check_mixed_types(df))

        # Check 6: Target imbalance
        if target_column and task_type == "classification":
            findings.extend(self._check_target_imbalance(df, target_column))

        # Check 7: Extreme values / outliers
        findings.extend(self._check_extreme_values(df))

        return findings

    def _check_missing_values(self, df: pd.DataFrame) -> list[AuditFinding]:
        findings = []

        missing_per_col = df.isna().mean()
        high_missing = missing_per_col[
            missing_per_col > MISSING_VALUE_WARNING_THRESHOLD
        ].sort_values(ascending=False)

        if len(high_missing) > 0:
            critical_cols = high_missing[
                high_missing > MISSING_VALUE_CRITICAL_THRESHOLD
            ]
            warning_cols = high_missing[
                (high_missing > MISSING_VALUE_WARNING_THRESHOLD) &
                (high_missing <= MISSING_VALUE_CRITICAL_THRESHOLD)
            ]

            if len(critical_cols) > 0:
                findings.append(self._create_finding(
                    severity=Severity.HIGH,
                    status=FindingStatus.CONFIRMED,
                    description=(
                        f"{len(critical_cols)} column(s) have >50% missing values."
                    ),
                    evidence={
                        "columns": {
                            col: f"{pct:.1%}"
                            for col, pct in critical_cols.items()
                        },
                    },
                    affected_columns=list(critical_cols.index),
                    explanation=(
                        "Columns with majority missing values provide little "
                        "information and may cause imputation to dominate. "
                        "Consider removing these columns."
                    ),
                    confidence=1.0,
                    recommendation=(
                        "Remove columns with >50% missing values or "
                        "investigate why data is missing."
                    ),
                    auto_correctable=True,
                ))

            if len(warning_cols) > 0:
                findings.append(self._create_finding(
                    severity=Severity.MEDIUM,
                    status=FindingStatus.CONFIRMED,
                    description=(
                        f"{len(warning_cols)} column(s) have 5–50% missing values."
                    ),
                    evidence={
                        "columns": {
                            col: f"{pct:.1%}"
                            for col, pct in warning_cols.items()
                        },
                    },
                    affected_columns=list(warning_cols.index),
                    explanation=(
                        "Moderate missing values can affect model performance. "
                        "Ensure imputation is done within the pipeline (after splitting)."
                    ),
                    confidence=1.0,
                    recommendation=(
                        "Use SimpleImputer inside a Pipeline. "
                        "Median for numeric, most_frequent for categorical."
                    ),
                    auto_correctable=False,
                ))

        return findings

    def _check_duplicates(self, df: pd.DataFrame) -> list[AuditFinding]:
        findings = []

        dup_count = int(df.duplicated().sum())
        dup_pct = dup_count / max(len(df), 1) * 100

        if dup_pct > 5:
            findings.append(self._create_finding(
                severity=Severity.MEDIUM,
                status=FindingStatus.CONFIRMED,
                description=(
                    f"Dataset has {dup_count} duplicate rows ({dup_pct:.1f}%)."
                ),
                evidence={
                    "duplicate_count": dup_count,
                    "duplicate_pct": round(dup_pct, 2),
                    "total_rows": len(df),
                },
                explanation=(
                    "Duplicate rows can bias training and inflate test metrics "
                    "if they end up in both train and test sets."
                ),
                confidence=1.0,
                recommendation=(
                    "Remove duplicates before splitting, unless they represent "
                    "legitimate repeated measurements."
                ),
                auto_correctable=True,
            ))
        elif dup_count > 0:
            findings.append(self._create_finding(
                severity=Severity.LOW,
                status=FindingStatus.INFORMATIONAL,
                description=f"Dataset has {dup_count} duplicate rows ({dup_pct:.1f}%).",
                evidence={
                    "duplicate_count": dup_count,
                    "duplicate_pct": round(dup_pct, 2),
                },
                explanation="Minor duplication detected.",
                confidence=1.0,
                recommendation="Review whether duplicates are expected.",
                auto_correctable=False,
            ))

        return findings

    def _check_constant_columns(
        self, df: pd.DataFrame, target_column: str
    ) -> list[AuditFinding]:
        findings = []

        constant_cols = [
            col for col in df.columns
            if col != target_column and df[col].nunique(dropna=True) <= 1
        ]

        if constant_cols:
            findings.append(self._create_finding(
                severity=Severity.MEDIUM,
                status=FindingStatus.CONFIRMED,
                description=(
                    f"{len(constant_cols)} constant/single-value column(s) found."
                ),
                evidence={
                    "columns": constant_cols,
                    "values": {
                        col: str(df[col].dropna().unique()[0])
                        if df[col].dropna().nunique() == 1
                        else "all_null"
                        for col in constant_cols
                    },
                },
                affected_columns=constant_cols,
                explanation=(
                    "Constant columns have no predictive power and add noise."
                ),
                confidence=1.0,
                recommendation="Remove constant columns before training.",
                auto_correctable=True,
            ))

        return findings

    def _check_infinite_values(self, df: pd.DataFrame) -> list[AuditFinding]:
        findings = []

        numeric_df = df.select_dtypes(include=[np.number])
        inf_cols = []

        for col in numeric_df.columns:
            if np.isinf(numeric_df[col]).any():
                inf_count = int(np.isinf(numeric_df[col]).sum())
                inf_cols.append((col, inf_count))

        if inf_cols:
            findings.append(self._create_finding(
                severity=Severity.HIGH,
                status=FindingStatus.CONFIRMED,
                description=(
                    f"{len(inf_cols)} column(s) contain infinite values."
                ),
                evidence={
                    "columns": {col: count for col, count in inf_cols},
                },
                affected_columns=[col for col, _ in inf_cols],
                explanation=(
                    "Infinite values will cause many models to fail or "
                    "produce meaningless results."
                ),
                confidence=1.0,
                recommendation=(
                    "Replace infinite values with NaN and impute, "
                    "or clip to reasonable bounds."
                ),
                auto_correctable=True,
            ))

        return findings

    def _check_mixed_types(self, df: pd.DataFrame) -> list[AuditFinding]:
        findings = []

        for col in df.select_dtypes(include=["object"]).columns:
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue

            # Check if values look numeric
            numeric_count = pd.to_numeric(non_null, errors="coerce").notna().sum()
            total = len(non_null)
            numeric_pct = numeric_count / total

            if 0.1 < numeric_pct < 0.9:
                findings.append(self._create_finding(
                    severity=Severity.MEDIUM,
                    status=FindingStatus.CONFIRMED,
                    description=(
                        f"Column '{col}' has mixed types: "
                        f"{numeric_pct:.0%} numeric, {1-numeric_pct:.0%} non-numeric."
                    ),
                    evidence={
                        "column": col,
                        "numeric_pct": round(numeric_pct * 100, 1),
                        "total_values": total,
                        "sample_non_numeric": [
                            str(v) for v in non_null[
                                pd.to_numeric(non_null, errors="coerce").isna()
                            ].head(5).values
                        ],
                    },
                    affected_columns=[col],
                    explanation=(
                        f"Column '{col}' contains a mix of numeric and "
                        "non-numeric values, which may indicate data quality issues."
                    ),
                    confidence=0.9,
                    recommendation="Clean this column: convert to numeric or categorical.",
                    auto_correctable=False,
                ))

        return findings

    def _check_target_imbalance(
        self, df: pd.DataFrame, target_column: str
    ) -> list[AuditFinding]:
        findings = []

        target = df[target_column].dropna()
        value_counts = target.value_counts(normalize=True)

        if len(value_counts) < 2:
            findings.append(self._create_finding(
                severity=Severity.CRITICAL,
                status=FindingStatus.CONFIRMED,
                description=f"Target '{target_column}' has only one class.",
                evidence={"unique_values": int(target.nunique())},
                affected_columns=[target_column],
                explanation="Classification requires at least 2 classes.",
                confidence=1.0,
                recommendation="Verify target column selection or use regression.",
                auto_correctable=False,
            ))
            return findings

        min_pct = float(value_counts.min()) * 100

        if min_pct < CLASS_IMBALANCE_THRESHOLD * 100:
            findings.append(self._create_finding(
                severity=Severity.MEDIUM,
                status=FindingStatus.CONFIRMED,
                description=(
                    f"Target is severely imbalanced — minority class: {min_pct:.1f}%."
                ),
                evidence={
                    "class_distribution": {
                        str(k): round(float(v) * 100, 2)
                        for k, v in value_counts.items()
                    },
                    "minority_class_pct": round(min_pct, 2),
                    "n_classes": len(value_counts),
                },
                affected_columns=[target_column],
                explanation=(
                    f"The minority class represents only {min_pct:.1f}% of the data. "
                    "Accuracy can be misleading — use F1, precision, recall, "
                    "or PR-AUC for evaluation."
                ),
                confidence=1.0,
                recommendation=(
                    "Use stratified splitting, class-weighted models, "
                    "and appropriate metrics (F1, PR-AUC)."
                ),
                auto_correctable=False,
            ))

        return findings

    def _check_extreme_values(self, df: pd.DataFrame) -> list[AuditFinding]:
        findings = []

        numeric_df = df.select_dtypes(include=[np.number])
        extreme_cols = []

        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if len(series) < 20:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - 3 * iqr
            upper = q3 + 3 * iqr
            n_extreme = int(((series < lower) | (series > upper)).sum())
            extreme_pct = n_extreme / len(series) * 100

            if extreme_pct > 5:
                extreme_cols.append((col, n_extreme, round(extreme_pct, 1)))

        if extreme_cols:
            findings.append(self._create_finding(
                severity=Severity.LOW,
                status=FindingStatus.INFORMATIONAL,
                description=(
                    f"{len(extreme_cols)} column(s) have >5% extreme values (3×IQR)."
                ),
                evidence={
                    "columns": {
                        col: {"count": count, "pct": pct}
                        for col, count, pct in extreme_cols
                    },
                },
                affected_columns=[col for col, _, _ in extreme_cols],
                explanation=(
                    "Extreme values (beyond 3× IQR) can disproportionately "
                    "influence model training."
                ),
                confidence=0.6,
                recommendation=(
                    "Consider using robust scaling or winsorization. "
                    "Verify extreme values are legitimate."
                ),
                auto_correctable=False,
            ))

        return findings

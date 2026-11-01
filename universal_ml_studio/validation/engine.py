"""
Validation Engine — 4-tier automatic dataset quality assessment.

Tiers
-----
1. File-Level        – size, format, encoding, readability
2. Structural        – row/column counts, duplicates, column name hygiene
3. Data Quality      – missing values, outliers, cardinality, type consistency
4. ML Readiness      – target column, class balance, leakage, trainability

Public API
----------
    ValidationEngine.validate(df, metadata, target_column=None)
        → ValidationResult (dataclass)
    ValidationEngine.save_report(result, dataset_id, session=None)
        → ValidationReport ORM object
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..config import (
    MIN_ROWS_FOR_ML,
    MIN_FEATURE_COLUMNS,
    HIGH_CARDINALITY_THRESHOLD,
    CONSTANT_COLUMN_NUNIQUE,
    MISSING_VALUE_WARNING_PCT,
    MISSING_VALUE_ERROR_PCT,
    CLASS_IMBALANCE_RATIO,
    MIN_SAMPLES_PER_CLASS,
    OUTLIER_IQR_MULTIPLIER,
    OUTLIER_ZSCORE_THRESHOLD,
    CORRELATION_LEAKAGE_THRESHOLD,
    MAX_UNIQUE_FOR_CLASSIFICATION,
    MAX_UPLOAD_SIZE_MB,
    SUPPORTED_EXTENSIONS,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CheckResult:
    """Single validation check outcome."""
    category: str          # file | structural | quality | ml_readiness
    check_name: str
    status: str            # passed | failed | warning
    message: str = ""
    affected_columns: list[str] = field(default_factory=list)
    affected_rows: int = 0
    suggestion: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Aggregated result of all validation checks."""
    status: str = "pending"             # valid | valid_with_warnings | invalid
    checks: list[CheckResult] = field(default_factory=list)
    is_trainable: bool = False
    summary: dict = field(default_factory=dict)

    @property
    def total_checks(self) -> int:
        return len(self.checks)

    @property
    def passed_checks(self) -> int:
        return sum(1 for c in self.checks if c.status == "passed")

    @property
    def failed_checks(self) -> int:
        return sum(1 for c in self.checks if c.status == "failed")

    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "warning")

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "is_trainable": self.is_trainable,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "warning_count": self.warning_count,
            "summary": self.summary,
            "checks": [
                {
                    "tier": c.category,
                    "check_name": c.check_name,
                    "status": c.status,
                    "message": c.message,
                    "affected_columns": c.affected_columns,
                    "affected_rows": c.affected_rows,
                    "suggestion": c.suggestion,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Validation Engine
# ═══════════════════════════════════════════════════════════════════════════════

class ValidationEngine:
    """Run a comprehensive 4-tier validation pipeline on a dataset."""

    # ── Main entry point ──────────────────────────────────────────────────

    @staticmethod
    def validate(
        df: pd.DataFrame,
        metadata: dict[str, Any],
        target_column: str | None = None,
    ) -> ValidationResult:
        """
        Run all 30+ checks and return a ValidationResult.

        Parameters
        ----------
        df : DataFrame to validate.
        metadata : Metadata dict from DatasetIngestion.extract_metadata().
        target_column : Optional target column for ML-readiness checks.
        """
        result = ValidationResult()

        # Tier 1: File-level checks
        ValidationEngine._tier1_file_checks(result, metadata)

        # Tier 2: Structural checks
        ValidationEngine._tier2_structural_checks(result, df, metadata)

        # Tier 3: Data quality checks
        ValidationEngine._tier3_quality_checks(result, df, metadata)

        # Tier 4: ML-readiness checks
        ValidationEngine._tier4_ml_readiness_checks(result, df, metadata, target_column)

        # Compute final status
        ValidationEngine._compute_final_status(result)

        logger.info(
            "Validation complete — %s (%d passed, %d warnings, %d failed)",
            result.status, result.passed_checks, result.warning_count, result.failed_checks,
        )
        return result

    # ── Save to database ──────────────────────────────────────────────────

    @staticmethod
    def save_report(
        result: ValidationResult,
        dataset_id: str,
        session=None,
    ):
        """Persist a ValidationResult to the database."""
        from ..database.models import ValidationReport, ValidationCheck
        from ..database.connection import get_session

        own_session = session is None
        if own_session:
            session = get_session()

        try:
            report = ValidationReport(
                dataset_id=dataset_id,
                status=result.status,
                total_checks=result.total_checks,
                passed_checks=result.passed_checks,
                failed_checks=result.failed_checks,
                warning_count=result.warning_count,
                is_trainable=result.is_trainable,
                summary=result.summary,
            )
            session.add(report)
            session.flush()  # get report.id

            for check in result.checks:
                vc = ValidationCheck(
                    report_id=report.id,
                    category=check.category,
                    check_name=check.check_name,
                    status=check.status,
                    message=check.message,
                    affected_columns=check.affected_columns or None,
                    affected_rows=check.affected_rows,
                    suggestion=check.suggestion,
                    details=check.details or None,
                )
                session.add(vc)

            if own_session:
                session.commit()
                session.expunge(report)

            return report

        except Exception:
            if own_session:
                session.rollback()
            raise
        finally:
            if own_session:
                session.close()

    # ═══════════════════════════════════════════════════════════════════════
    #  TIER 1 — File-level checks
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _tier1_file_checks(result: ValidationResult, metadata: dict):
        """Checks: file format, file size, encoding / readability."""

        # 1.1 — File format
        fmt = metadata.get("file_format", "")
        ext = f".{fmt}" if fmt else ""
        if ext and ext in SUPPORTED_EXTENSIONS:
            result.checks.append(CheckResult(
                category="file",
                check_name="file_format_supported",
                status="passed",
                message=f"File format '.{fmt}' is supported.",
            ))
        elif ext:
            result.checks.append(CheckResult(
                category="file",
                check_name="file_format_supported",
                status="failed",
                message=f"File format '.{fmt}' is not supported.",
                suggestion=f"Convert the file to one of: {SUPPORTED_EXTENSIONS}",
            ))
        # If no format info, skip this check (metadata may be from DataFrame-only)

        # 1.2 — File size
        size_bytes = metadata.get("file_size_bytes", 0)
        if size_bytes > 0:
            size_mb = size_bytes / (1024 * 1024)
            if size_mb <= MAX_UPLOAD_SIZE_MB:
                result.checks.append(CheckResult(
                    category="file",
                    check_name="file_size_within_limit",
                    status="passed",
                    message=f"File size ({size_mb:.1f} MB) is within the {MAX_UPLOAD_SIZE_MB} MB limit.",
                ))
            else:
                result.checks.append(CheckResult(
                    category="file",
                    check_name="file_size_within_limit",
                    status="failed",
                    message=f"File size ({size_mb:.1f} MB) exceeds the {MAX_UPLOAD_SIZE_MB} MB limit.",
                    suggestion="Reduce the file size or contact an administrator.",
                ))

        # 1.3 — Readability (if we got here, the file was already read)
        result.checks.append(CheckResult(
            category="file",
            check_name="file_readable",
            status="passed",
            message="File was read successfully.",
        ))

    # ═══════════════════════════════════════════════════════════════════════
    #  TIER 2 — Structural checks
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _tier2_structural_checks(result: ValidationResult, df: pd.DataFrame, metadata: dict):
        """Checks: row count, column count, duplicated columns, empty columns,
        column-name hygiene, duplicate rows."""

        num_rows, num_cols = df.shape

        # 2.1 — Minimum row count
        if num_rows >= MIN_ROWS_FOR_ML:
            result.checks.append(CheckResult(
                category="structural",
                check_name="minimum_row_count",
                status="passed",
                message=f"Dataset has {num_rows} rows (minimum: {MIN_ROWS_FOR_ML}).",
            ))
        else:
            result.checks.append(CheckResult(
                category="structural",
                check_name="minimum_row_count",
                status="failed",
                message=f"Dataset has only {num_rows} rows (minimum: {MIN_ROWS_FOR_ML}).",
                suggestion="Collect more data before attempting ML training.",
            ))

        # 2.2 — Minimum column count
        if num_cols >= 2:  # at least target + 1 feature
            result.checks.append(CheckResult(
                category="structural",
                check_name="minimum_column_count",
                status="passed",
                message=f"Dataset has {num_cols} columns.",
            ))
        else:
            result.checks.append(CheckResult(
                category="structural",
                check_name="minimum_column_count",
                status="failed",
                message=f"Dataset has only {num_cols} column(s). At least 2 required.",
                suggestion="Add feature columns alongside the target column.",
            ))

        # 2.3 — Duplicate column names
        dup_cols = [c for c in df.columns if list(df.columns).count(c) > 1]
        if not dup_cols:
            result.checks.append(CheckResult(
                category="structural",
                check_name="no_duplicate_column_names",
                status="passed",
                message="All column names are unique.",
            ))
        else:
            result.checks.append(CheckResult(
                category="structural",
                check_name="no_duplicate_column_names",
                status="failed",
                message=f"Found duplicate column names: {list(set(dup_cols))}",
                affected_columns=list(set(dup_cols)),
                suggestion="Rename columns to be unique before proceeding.",
            ))

        # 2.4 — Entirely empty columns (100% null)
        empty_cols = [c for c in df.columns if df[c].isnull().all()]
        if not empty_cols:
            result.checks.append(CheckResult(
                category="structural",
                check_name="no_empty_columns",
                status="passed",
                message="No entirely empty columns found.",
            ))
        else:
            result.checks.append(CheckResult(
                category="structural",
                check_name="no_empty_columns",
                status="warning",
                message=f"{len(empty_cols)} column(s) are entirely empty.",
                affected_columns=empty_cols,
                suggestion="Drop these columns or fill with meaningful values.",
            ))

        # 2.5 — Constant columns (only 1 unique value)
        constant_cols = [
            c for c in df.columns
            if df[c].nunique(dropna=True) <= CONSTANT_COLUMN_NUNIQUE
            and not df[c].isnull().all()
        ]
        if not constant_cols:
            result.checks.append(CheckResult(
                category="structural",
                check_name="no_constant_columns",
                status="passed",
                message="No constant columns detected.",
            ))
        else:
            result.checks.append(CheckResult(
                category="structural",
                check_name="no_constant_columns",
                status="warning",
                message=f"{len(constant_cols)} column(s) have only 1 unique value.",
                affected_columns=constant_cols,
                suggestion="Constant columns provide no information — consider dropping them.",
            ))

        # 2.6 — Column name hygiene (whitespace, special characters)
        bad_names = [
            c for c in df.columns
            if c != c.strip() or "\n" in str(c) or "\t" in str(c)
        ]
        if not bad_names:
            result.checks.append(CheckResult(
                category="structural",
                check_name="column_name_hygiene",
                status="passed",
                message="All column names are clean.",
            ))
        else:
            result.checks.append(CheckResult(
                category="structural",
                check_name="column_name_hygiene",
                status="warning",
                message=f"{len(bad_names)} column(s) have leading/trailing whitespace or control characters.",
                affected_columns=bad_names,
                suggestion="Strip whitespace from column names.",
            ))

        # 2.7 — Duplicate rows
        dup_row_count = int(df.duplicated().sum())
        if dup_row_count == 0:
            result.checks.append(CheckResult(
                category="structural",
                check_name="no_duplicate_rows",
                status="passed",
                message="No duplicate rows found.",
            ))
        else:
            dup_pct = round(dup_row_count / num_rows * 100, 2)
            status = "warning" if dup_pct < 20 else "failed"
            result.checks.append(CheckResult(
                category="structural",
                check_name="no_duplicate_rows",
                status=status,
                message=f"{dup_row_count} duplicate rows ({dup_pct}%).",
                affected_rows=dup_row_count,
                suggestion="Consider dropping duplicate rows to avoid bias.",
            ))

    # ═══════════════════════════════════════════════════════════════════════
    #  TIER 3 — Data Quality checks
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _tier3_quality_checks(result: ValidationResult, df: pd.DataFrame, metadata: dict):
        """Checks: missing values, outliers, high cardinality, type consistency,
        mixed types, infinite values."""

        num_rows = len(df)

        # 3.1 — Overall missing value percentage
        total_cells = num_rows * len(df.columns)
        total_missing = int(df.isnull().sum().sum())
        if total_cells > 0:
            missing_pct = round(total_missing / total_cells * 100, 2)
        else:
            missing_pct = 0.0

        if missing_pct == 0:
            result.checks.append(CheckResult(
                category="quality",
                check_name="overall_missing_values",
                status="passed",
                message="No missing values in the dataset.",
            ))
        elif missing_pct < MISSING_VALUE_WARNING_PCT:
            result.checks.append(CheckResult(
                category="quality",
                check_name="overall_missing_values",
                status="passed",
                message=f"Missing values: {missing_pct}% (below {MISSING_VALUE_WARNING_PCT}% threshold).",
                details={"missing_pct": missing_pct, "total_missing": total_missing},
            ))
        elif missing_pct < MISSING_VALUE_ERROR_PCT:
            result.checks.append(CheckResult(
                category="quality",
                check_name="overall_missing_values",
                status="warning",
                message=f"Missing values: {missing_pct}% — may affect model quality.",
                details={"missing_pct": missing_pct, "total_missing": total_missing},
                suggestion="Impute or drop columns/rows with excessive missing values.",
            ))
        else:
            result.checks.append(CheckResult(
                category="quality",
                check_name="overall_missing_values",
                status="failed",
                message=f"Missing values: {missing_pct}% — too many for reliable ML.",
                details={"missing_pct": missing_pct, "total_missing": total_missing},
                suggestion="This dataset is mostly empty. Collect better data.",
            ))

        # 3.2 — Per-column missing value warnings
        columns_with_high_missing = []
        for col in df.columns:
            col_missing_pct = df[col].isnull().mean() * 100
            if col_missing_pct > MISSING_VALUE_WARNING_PCT:
                columns_with_high_missing.append(
                    {"column": col, "missing_pct": round(col_missing_pct, 2)}
                )
        if columns_with_high_missing:
            affected = [c["column"] for c in columns_with_high_missing]
            result.checks.append(CheckResult(
                category="quality",
                check_name="per_column_missing_values",
                status="warning",
                message=f"{len(columns_with_high_missing)} column(s) have >{MISSING_VALUE_WARNING_PCT}% missing values.",
                affected_columns=affected,
                suggestion="Consider imputing or dropping these columns.",
                details={"columns": columns_with_high_missing},
            ))
        else:
            result.checks.append(CheckResult(
                category="quality",
                check_name="per_column_missing_values",
                status="passed",
                message=f"No columns exceed {MISSING_VALUE_WARNING_PCT}% missing values.",
            ))

        # 3.3 — Outlier detection (IQR method on numerical columns)
        numerical_cols = metadata.get("numerical_columns", [])
        outlier_columns = []
        for col in numerical_cols:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) < 10:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - OUTLIER_IQR_MULTIPLIER * iqr
            upper = q3 + OUTLIER_IQR_MULTIPLIER * iqr
            outlier_count = int(((series < lower) | (series > upper)).sum())
            outlier_pct = round(outlier_count / len(series) * 100, 2)
            if outlier_pct > 5:
                outlier_columns.append({
                    "column": col, "outlier_count": outlier_count,
                    "outlier_pct": outlier_pct,
                })

        if outlier_columns:
            affected = [c["column"] for c in outlier_columns]
            result.checks.append(CheckResult(
                category="quality",
                check_name="outlier_detection",
                status="warning",
                message=f"{len(outlier_columns)} column(s) have >5% outliers (IQR method).",
                affected_columns=affected,
                suggestion="Consider capping, transforming, or investigating outliers.",
                details={"columns": outlier_columns},
            ))
        else:
            result.checks.append(CheckResult(
                category="quality",
                check_name="outlier_detection",
                status="passed",
                message="No columns with excessive outliers detected.",
            ))

        # 3.4 — High cardinality categorical columns
        categorical_cols = metadata.get("categorical_columns", [])
        high_card_cols = []
        for col in categorical_cols:
            if col not in df.columns:
                continue
            nunique = df[col].nunique(dropna=True)
            if nunique > HIGH_CARDINALITY_THRESHOLD:
                high_card_cols.append({"column": col, "nunique": nunique})

        if high_card_cols:
            affected = [c["column"] for c in high_card_cols]
            result.checks.append(CheckResult(
                category="quality",
                check_name="high_cardinality",
                status="warning",
                message=f"{len(high_card_cols)} categorical column(s) have >{HIGH_CARDINALITY_THRESHOLD} unique values.",
                affected_columns=affected,
                suggestion="Consider target encoding, hashing, or dropping high-cardinality columns.",
                details={"columns": high_card_cols},
            ))
        else:
            result.checks.append(CheckResult(
                category="quality",
                check_name="high_cardinality",
                status="passed",
                message="No high-cardinality categorical columns detected.",
            ))

        # 3.5 — Infinite values in numerical columns
        inf_cols = []
        for col in numerical_cols:
            if col not in df.columns:
                continue
            inf_count = int(np.isinf(df[col].dropna().values).sum()) if df[col].dtype != object else 0
            if inf_count > 0:
                inf_cols.append({"column": col, "inf_count": inf_count})

        if inf_cols:
            affected = [c["column"] for c in inf_cols]
            result.checks.append(CheckResult(
                category="quality",
                check_name="no_infinite_values",
                status="warning",
                message=f"{len(inf_cols)} column(s) contain infinite values.",
                affected_columns=affected,
                suggestion="Replace infinite values with NaN or a finite cap.",
                details={"columns": inf_cols},
            ))
        else:
            result.checks.append(CheckResult(
                category="quality",
                check_name="no_infinite_values",
                status="passed",
                message="No infinite values found in numerical columns.",
            ))

        # 3.6 — Mixed types detection (object columns with mixed numeric/string)
        mixed_type_cols = []
        for col in df.select_dtypes(include=["object"]).columns:
            sample = df[col].dropna().head(100)
            if sample.empty:
                continue
            numeric_count = pd.to_numeric(sample, errors="coerce").notna().sum()
            string_count = len(sample) - numeric_count
            if numeric_count > 0 and string_count > 0:
                ratio = min(numeric_count, string_count) / len(sample)
                if ratio > 0.1:  # more than 10% are the minority type
                    mixed_type_cols.append(col)

        if mixed_type_cols:
            result.checks.append(CheckResult(
                category="quality",
                check_name="no_mixed_types",
                status="warning",
                message=f"{len(mixed_type_cols)} column(s) contain mixed numeric and string values.",
                affected_columns=mixed_type_cols,
                suggestion="Standardise column types — convert or separate values.",
            ))
        else:
            result.checks.append(CheckResult(
                category="quality",
                check_name="no_mixed_types",
                status="passed",
                message="No columns with mixed numeric/string types detected.",
            ))

        # 3.7 — Whitespace-only strings
        whitespace_cols = []
        for col in df.select_dtypes(include=["object", "string"]).columns:
            ws_count = int(df[col].dropna().apply(
                lambda x: isinstance(x, str) and len(x.strip()) == 0
            ).sum())
            if ws_count > 0:
                whitespace_cols.append({"column": col, "count": ws_count})

        if whitespace_cols:
            affected = [c["column"] for c in whitespace_cols]
            result.checks.append(CheckResult(
                category="quality",
                check_name="no_whitespace_only_strings",
                status="warning",
                message=f"{len(whitespace_cols)} column(s) contain whitespace-only strings.",
                affected_columns=affected,
                suggestion="Replace whitespace-only strings with NaN.",
                details={"columns": whitespace_cols},
            ))
        else:
            result.checks.append(CheckResult(
                category="quality",
                check_name="no_whitespace_only_strings",
                status="passed",
                message="No whitespace-only string values found.",
            ))

    # ═══════════════════════════════════════════════════════════════════════
    #  TIER 4 — ML Readiness checks
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _tier4_ml_readiness_checks(
        result: ValidationResult,
        df: pd.DataFrame,
        metadata: dict,
        target_column: str | None,
    ):
        """Checks: target existence, class balance, leakage, feature count,
        label noise, target missing values."""

        numerical_cols = metadata.get("numerical_columns", [])

        # 4.1 — Target column existence
        if target_column is None:
            result.checks.append(CheckResult(
                category="ml_readiness",
                check_name="target_column_specified",
                status="warning",
                message="No target column specified — ML readiness checks are limited.",
                suggestion="Specify a target column for full validation.",
            ))
            # Still run feature-count check
            ValidationEngine._check_feature_count(result, df, None)
            return

        if target_column not in df.columns:
            result.checks.append(CheckResult(
                category="ml_readiness",
                check_name="target_column_exists",
                status="failed",
                message=f"Target column '{target_column}' does not exist in the dataset.",
                suggestion=f"Available columns: {df.columns.tolist()[:20]}",
            ))
            return

        result.checks.append(CheckResult(
            category="ml_readiness",
            check_name="target_column_exists",
            status="passed",
            message=f"Target column '{target_column}' found in dataset.",
        ))

        # 4.2 — Target column missing values
        target_missing = int(df[target_column].isnull().sum())
        if target_missing == 0:
            result.checks.append(CheckResult(
                category="ml_readiness",
                check_name="target_no_missing",
                status="passed",
                message=f"Target column '{target_column}' has no missing values.",
            ))
        else:
            pct = round(target_missing / len(df) * 100, 2)
            result.checks.append(CheckResult(
                category="ml_readiness",
                check_name="target_no_missing",
                status="failed" if pct > 10 else "warning",
                message=f"Target column has {target_missing} missing values ({pct}%).",
                affected_columns=[target_column],
                affected_rows=target_missing,
                suggestion="Drop rows with missing target values before training.",
            ))

        # 4.3 — Feature column count (at least MIN_FEATURE_COLUMNS besides target)
        ValidationEngine._check_feature_count(result, df, target_column)

        # 4.4 — Determine if likely classification or regression
        target_nunique = df[target_column].nunique(dropna=True)
        is_classification = (
            target_nunique <= MAX_UNIQUE_FOR_CLASSIFICATION
            or df[target_column].dtype == object
            or df[target_column].dtype.name in ("category", "bool")
        )

        result.checks.append(CheckResult(
            category="ml_readiness",
            check_name="task_type_inference",
            status="passed",
            message=f"Inferred task type: {'classification' if is_classification else 'regression'} "
                    f"({target_nunique} unique target values).",
            details={"inferred_task": "classification" if is_classification else "regression",
                     "target_nunique": target_nunique},
        ))

        # 4.5 — Class balance (classification only)
        if is_classification:
            value_counts = df[target_column].value_counts(dropna=True)
            if len(value_counts) >= 2:
                majority = value_counts.iloc[0]
                minority = value_counts.iloc[-1]
                ratio = round(majority / max(minority, 1), 2)

                if ratio <= CLASS_IMBALANCE_RATIO:
                    result.checks.append(CheckResult(
                        category="ml_readiness",
                        check_name="class_balance",
                        status="passed",
                        message=f"Class balance ratio: {ratio}:1 (threshold: {CLASS_IMBALANCE_RATIO}:1).",
                        details={"ratio": ratio, "class_counts": value_counts.to_dict()},
                    ))
                else:
                    result.checks.append(CheckResult(
                        category="ml_readiness",
                        check_name="class_balance",
                        status="warning",
                        message=f"Class imbalance detected — ratio {ratio}:1 exceeds {CLASS_IMBALANCE_RATIO}:1.",
                        details={"ratio": ratio, "class_counts": value_counts.to_dict()},
                        suggestion="Consider SMOTE, class weights, or undersampling.",
                    ))

            # 4.6 — Minimum samples per class
            small_classes = [
                str(cls) for cls, cnt in value_counts.items()
                if cnt < MIN_SAMPLES_PER_CLASS
            ]
            if small_classes:
                result.checks.append(CheckResult(
                    category="ml_readiness",
                    check_name="min_samples_per_class",
                    status="warning",
                    message=f"{len(small_classes)} class(es) have fewer than {MIN_SAMPLES_PER_CLASS} samples.",
                    details={"small_classes": small_classes},
                    suggestion="Collect more samples or merge rare classes.",
                ))
            else:
                result.checks.append(CheckResult(
                    category="ml_readiness",
                    check_name="min_samples_per_class",
                    status="passed",
                    message=f"All classes have at least {MIN_SAMPLES_PER_CLASS} samples.",
                ))

        # 4.7 — Feature-target correlation leakage (numerical only)
        if target_column in numerical_cols and len(numerical_cols) > 1:
            leaky_features = []
            for col in numerical_cols:
                if col == target_column or col not in df.columns:
                    continue
                try:
                    corr = abs(df[col].corr(df[target_column]))
                    if not math.isnan(corr) and corr >= CORRELATION_LEAKAGE_THRESHOLD:
                        leaky_features.append({"column": col, "correlation": round(corr, 4)})
                except Exception:
                    continue

            if leaky_features:
                affected = [c["column"] for c in leaky_features]
                result.checks.append(CheckResult(
                    category="ml_readiness",
                    check_name="no_target_leakage",
                    status="warning",
                    message=f"{len(leaky_features)} feature(s) have suspiciously high correlation with target.",
                    affected_columns=affected,
                    suggestion="Investigate — these features may leak the target variable.",
                    details={"columns": leaky_features},
                ))
            else:
                result.checks.append(CheckResult(
                    category="ml_readiness",
                    check_name="no_target_leakage",
                    status="passed",
                    message="No features with suspiciously high target correlation detected.",
                ))

        # 4.8 — Target variance (regression only — zero variance is useless)
        if not is_classification and target_column in numerical_cols:
            target_std = df[target_column].dropna().std()
            if target_std == 0:
                result.checks.append(CheckResult(
                    category="ml_readiness",
                    check_name="target_has_variance",
                    status="failed",
                    message="Target column has zero variance — nothing to predict.",
                    affected_columns=[target_column],
                    suggestion="Check if the target column is correct.",
                ))
            else:
                result.checks.append(CheckResult(
                    category="ml_readiness",
                    check_name="target_has_variance",
                    status="passed",
                    message=f"Target column has std={round(float(target_std), 4)}.",
                ))

    # ── Helper: feature count check ────────────────────────────────────────

    @staticmethod
    def _check_feature_count(result: ValidationResult, df: pd.DataFrame, target_column: str | None):
        feature_count = len(df.columns) - (1 if target_column else 0)
        if feature_count >= MIN_FEATURE_COLUMNS:
            result.checks.append(CheckResult(
                category="ml_readiness",
                check_name="sufficient_features",
                status="passed",
                message=f"{feature_count} feature column(s) available (minimum: {MIN_FEATURE_COLUMNS}).",
            ))
        else:
            result.checks.append(CheckResult(
                category="ml_readiness",
                check_name="sufficient_features",
                status="failed",
                message=f"Only {feature_count} feature column(s) — need at least {MIN_FEATURE_COLUMNS}.",
                suggestion="Add more feature columns.",
            ))

    # ═══════════════════════════════════════════════════════════════════════
    #  Final status computation
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_final_status(result: ValidationResult):
        """Determine overall status and trainability from individual checks."""
        if result.failed_checks > 0:
            result.status = "invalid"
            result.is_trainable = False
        elif result.warning_count > 0:
            result.status = "valid_with_warnings"
            result.is_trainable = True
        else:
            result.status = "valid"
            result.is_trainable = True

        result.summary = {
            "status": result.status,
            "total_checks": result.total_checks,
            "passed": result.passed_checks,
            "warnings": result.warning_count,
            "failed": result.failed_checks,
            "is_trainable": result.is_trainable,
            "categories": {
                cat: {
                    "total": sum(1 for c in result.checks if c.category == cat),
                    "passed": sum(1 for c in result.checks if c.category == cat and c.status == "passed"),
                    "warnings": sum(1 for c in result.checks if c.category == cat and c.status == "warning"),
                    "failed": sum(1 for c in result.checks if c.category == cat and c.status == "failed"),
                }
                for cat in ("file", "structural", "quality", "ml_readiness")
            },
        }

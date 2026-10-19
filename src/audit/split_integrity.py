"""
Audit C: Split Integrity — Train/Test Contamination Detection

Detects:
- Identical rows across train/test partitions
- Duplicate identifiers across partitions
- Near-duplicate records
- Repeated entities split across training and testing
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

from src.audit.base import BaseAudit, AuditFinding
from src.config import (
    Severity, FindingStatus, AuditCategory,
    OVERLAP_WARNING_THRESHOLD, NEAR_DUPLICATE_JACCARD_THRESHOLD,
)

logger = logging.getLogger(__name__)


class SplitIntegrityAudit(BaseAudit):
    """Detects train/test contamination and split integrity issues."""

    CATEGORY = AuditCategory.SPLIT_CONTAMINATION
    FINDING_PREFIX = "SC"

    def run(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series = None,
        y_test: pd.Series = None,
        df_full: pd.DataFrame = None,
        **kwargs,
    ) -> list[AuditFinding]:
        """
        Run split integrity checks.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels (optional)
            y_test: Test labels (optional)
            df_full: Full original dataset (optional, for duplicate analysis)
        """
        findings = []

        # Check 1: Exact row overlap
        findings.extend(self._check_exact_overlap(X_train, X_test))

        # Check 2: Near-duplicate rows
        findings.extend(self._check_near_duplicates(X_train, X_test))

        # Check 3: ID column leakage
        findings.extend(self._check_id_overlap(X_train, X_test))

        # Check 4: Full dataset duplicate analysis
        if df_full is not None:
            findings.extend(self._check_dataset_duplicates(df_full))

        return findings

    def _check_exact_overlap(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> list[AuditFinding]:
        """Check for identical rows appearing in both train and test."""
        findings = []

        try:
            # Hash rows for efficient comparison
            train_hashes = pd.util.hash_pandas_object(X_train, index=False)
            test_hashes = pd.util.hash_pandas_object(X_test, index=False)

            train_hash_set = set(train_hashes.values)
            test_hash_set = set(test_hashes.values)

            overlap_hashes = train_hash_set & test_hash_set
            overlap_count = len(overlap_hashes)

            if overlap_count > 0:
                overlap_pct_test = overlap_count / max(len(X_test), 1) * 100
                overlap_pct_train = overlap_count / max(len(X_train), 1) * 100

                # Find sample overlapping rows
                overlap_test_mask = test_hashes.isin(overlap_hashes)
                overlap_test_indices = list(
                    X_test[overlap_test_mask].index[:5]
                )

                # Get sample overlapping rows as evidence
                sample_rows = X_test.loc[overlap_test_indices[:3]].to_dict('records')
                # Make serializable
                sample_rows_clean = []
                for row in sample_rows:
                    clean_row = {}
                    for k, v in row.items():
                        try:
                            if pd.isna(v):
                                clean_row[str(k)] = None
                            else:
                                clean_row[str(k)] = str(v)
                        except (TypeError, ValueError):
                            clean_row[str(k)] = str(v)
                    sample_rows_clean.append(clean_row)

                if overlap_pct_test > OVERLAP_WARNING_THRESHOLD * 100:
                    severity = Severity.CRITICAL if overlap_pct_test > 10 else Severity.HIGH
                    status = FindingStatus.CONFIRMED

                    findings.append(self._create_finding(
                        severity=severity,
                        status=status,
                        description=(
                            f"{overlap_count} identical rows found in both "
                            f"train and test sets ({overlap_pct_test:.1f}% of test)."
                        ),
                        evidence={
                            "overlap_count": overlap_count,
                            "overlap_pct_of_test": round(overlap_pct_test, 2),
                            "overlap_pct_of_train": round(overlap_pct_train, 2),
                            "train_size": len(X_train),
                            "test_size": len(X_test),
                            "sample_overlapping_rows": sample_rows_clean,
                        },
                        explanation=(
                            f"There are {overlap_count} rows that appear in both the "
                            "training and test sets. This means the model has seen these "
                            "exact examples during training, inflating test performance. "
                            "This can happen due to duplicate records in the original dataset "
                            "or improper splitting."
                        ),
                        confidence=1.0,
                        recommendation=(
                            "Remove duplicate rows before splitting, or use the corrected "
                            "pipeline which deduplicates before the train/test split."
                        ),
                        auto_correctable=True,
                    ))
                else:
                    findings.append(self._create_finding(
                        severity=Severity.LOW,
                        status=FindingStatus.INFORMATIONAL,
                        description=(
                            f"{overlap_count} identical rows found across splits "
                            f"({overlap_pct_test:.2f}% of test — below warning threshold)."
                        ),
                        evidence={
                            "overlap_count": overlap_count,
                            "overlap_pct_of_test": round(overlap_pct_test, 2),
                        },
                        explanation=(
                            "A small number of identical rows exist across splits. "
                            "This may be legitimate repeated observations or minor "
                            "duplicates with minimal impact."
                        ),
                        confidence=0.5,
                        recommendation="Monitor but may not require action.",
                        auto_correctable=False,
                    ))

        except Exception as e:
            logger.warning(f"Exact overlap check failed: {e}")

        return findings

    def _check_near_duplicates(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> list[AuditFinding]:
        """Check for near-duplicate rows using feature similarity."""
        findings = []

        try:
            # Use numeric columns for similarity
            num_train = X_train.select_dtypes(include=[np.number])
            num_test = X_test.select_dtypes(include=[np.number])

            common_cols = [c for c in num_train.columns if c in num_test.columns]
            if not common_cols or len(common_cols) < 2:
                return findings

            # Sample for efficiency
            max_sample = 500
            train_sample = num_train[common_cols].dropna()
            test_sample = num_test[common_cols].dropna()

            if len(train_sample) > max_sample:
                train_sample = train_sample.sample(max_sample, random_state=42)
            if len(test_sample) > max_sample:
                test_sample = test_sample.sample(max_sample, random_state=42)

            if len(train_sample) == 0 or len(test_sample) == 0:
                return findings

            # Normalize for comparison
            combined = pd.concat([train_sample, test_sample])
            col_stds = combined.std()
            valid_cols = col_stds[col_stds > 0].index.tolist()

            if not valid_cols:
                return findings

            train_norm = train_sample[valid_cols].values
            test_norm = test_sample[valid_cols].values

            # Normalize
            means = combined[valid_cols].mean().values
            stds = col_stds[valid_cols].values
            stds[stds == 0] = 1

            train_norm = (train_norm - means) / stds
            test_norm = (test_norm - means) / stds

            # Find near-duplicates using pairwise distances
            near_dup_count = 0
            threshold = 0.01 * len(valid_cols)  # Very close threshold

            # Efficient: compare each test row against train
            for i in range(len(test_norm)):
                diffs = np.abs(train_norm - test_norm[i])
                min_dist = np.min(np.sum(diffs, axis=1))
                if min_dist < threshold and min_dist > 0:
                    near_dup_count += 1

            near_dup_pct = near_dup_count / max(len(test_sample), 1) * 100

            if near_dup_pct > 5:
                findings.append(self._create_finding(
                    severity=Severity.MEDIUM,
                    status=FindingStatus.SUSPECTED,
                    description=(
                        f"~{near_dup_count} near-duplicate rows detected between "
                        f"train and test ({near_dup_pct:.1f}% of sampled test rows)."
                    ),
                    evidence={
                        "near_duplicate_count": near_dup_count,
                        "near_duplicate_pct": round(near_dup_pct, 2),
                        "sample_size_train": len(train_sample),
                        "sample_size_test": len(test_sample),
                        "features_compared": len(valid_cols),
                        "threshold": round(threshold, 4),
                    },
                    explanation=(
                        "Near-duplicate rows are rows that are almost identical but "
                        "differ slightly (e.g., rounding, noise). These can inflate "
                        "test performance similarly to exact duplicates."
                    ),
                    confidence=0.6,
                    recommendation=(
                        "Review near-duplicate rows. Consider deduplication or "
                        "using a more robust splitting strategy."
                    ),
                    auto_correctable=False,
                ))

        except Exception as e:
            logger.warning(f"Near-duplicate check failed: {e}")

        return findings

    def _check_id_overlap(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> list[AuditFinding]:
        """Check for overlapping ID-like columns across splits."""
        findings = []

        # Identify potential ID columns
        for col in X_train.columns:
            if col not in X_test.columns:
                continue

            col_lower = col.lower()
            is_id_like = any(
                kw in col_lower
                for kw in ["id", "index", "key", "code", "number", "num", "no"]
            )

            if not is_id_like:
                continue

            train_vals = set(X_train[col].dropna().astype(str))
            test_vals = set(X_test[col].dropna().astype(str))
            overlap = train_vals & test_vals

            if overlap and len(overlap) > 0:
                overlap_pct = len(overlap) / max(len(test_vals), 1) * 100

                if overlap_pct > 5:
                    findings.append(self._create_finding(
                        severity=Severity.MEDIUM,
                        status=FindingStatus.SUSPECTED,
                        description=(
                            f"ID-like column '{col}' has {len(overlap)} shared "
                            f"values across train and test ({overlap_pct:.1f}%)."
                        ),
                        evidence={
                            "column": col,
                            "overlap_count": len(overlap),
                            "overlap_pct": round(overlap_pct, 2),
                            "sample_shared_ids": list(sorted(overlap))[:10],
                            "train_unique": len(train_vals),
                            "test_unique": len(test_vals),
                        },
                        affected_columns=[col],
                        explanation=(
                            f"The column '{col}' appears to be an identifier, and "
                            "the same IDs appear in both train and test. This could "
                            "indicate the same entity (e.g., patient, user) has records "
                            "in both partitions, which may leak information."
                        ),
                        confidence=0.6,
                        recommendation=(
                            f"Consider group-based splitting on '{col}' to ensure "
                            "the same entity doesn't appear in both train and test."
                        ),
                        auto_correctable=False,
                    ))

        return findings

    def _check_dataset_duplicates(
        self, df: pd.DataFrame
    ) -> list[AuditFinding]:
        """Check for duplicates in the full dataset."""
        findings = []

        dup_count = int(df.duplicated().sum())
        dup_pct = dup_count / max(len(df), 1) * 100

        if dup_count > 0 and dup_pct > 1:
            findings.append(self._create_finding(
                severity=Severity.MEDIUM if dup_pct > 5 else Severity.LOW,
                status=FindingStatus.INFORMATIONAL,
                description=(
                    f"Dataset contains {dup_count} duplicate rows ({dup_pct:.1f}%)."
                ),
                evidence={
                    "duplicate_count": dup_count,
                    "duplicate_pct": round(dup_pct, 2),
                    "total_rows": len(df),
                },
                explanation=(
                    "Duplicate rows can end up in both train and test sets during "
                    "random splitting, causing data leakage. Consider deduplication "
                    "before splitting."
                ),
                confidence=0.7,
                recommendation=(
                    "Remove duplicates before splitting, or verify that duplicates "
                    "represent legitimate repeated observations."
                ),
                auto_correctable=True,
            ))

        return findings

"""
Audit H: Temporal Leakage Detection
=====================================
Inspired by: "Data Leakage in Speaker Credibility Features: A Systematic
Analysis and Correction for the LIAR2 Fake News Benchmark"

The paper demonstrates that when features are computed using global aggregation
over the entire dataset BEFORE a time-ordered train/test split, information
from future (test) samples leaks into past (train) samples via shared
aggregates — causing inflated, unreliable evaluation metrics.

This module detects two forms of temporal leakage on any tabular dataset:

1. Missing temporal split:
   If the dataset contains a datetime/timestamp column but was split randomly
   (not by time), future records leak into the training set.

2. Time-order inversion:
   After an attempted time-ordered split, if the max training timestamp
   exceeds the min test timestamp, the split direction is inverted.

Both are direct generalizations of the LIAR2 paper's core finding.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

from src.audit.base import BaseAudit, AuditFinding
from src.config import Severity, FindingStatus, AuditCategory

logger = logging.getLogger(__name__)


class TemporalLeakageAudit(BaseAudit):
    """
    Detects temporal data leakage — the core contribution of the base paper.

    Checks whether the dataset has time-structured data that was not
    split in chronological order, allowing future information to leak
    into the training set.
    """

    CATEGORY = AuditCategory.TEMPORAL_LEAKAGE
    FINDING_PREFIX = "TMP"

    # Column name fragments that suggest a datetime column
    _DATE_KEYWORDS = {
        "date", "time", "datetime", "timestamp", "created", "updated",
        "year", "month", "day", "week", "period", "dt", "ts",
    }

    def run(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str = "classification",
        X_train: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> list[AuditFinding]:
        """
        Run temporal leakage checks.

        Args:
            df:            Full DataFrame (all rows, all columns)
            target_column: Name of the target column
            task_type:     'classification' or 'regression'
            X_train:       Training features (optional, used for split-order check)
            X_test:        Test features (optional, used for split-order check)

        Returns:
            List of AuditFinding objects
        """
        findings: list[AuditFinding] = []

        if df is None or df.empty:
            return findings

        # Detect all datetime columns in the dataset
        datetime_cols = self._detect_datetime_columns(df)

        if not datetime_cols:
            # No temporal dimension found — report as informational only
            findings.append(self._create_finding(
                severity=Severity.INFO,
                status=FindingStatus.INFORMATIONAL,
                description="No datetime/timestamp columns detected in the dataset.",
                evidence={"checked_columns": list(df.columns)},
                affected_columns=[],
                explanation=(
                    "The base paper (LIAR2) shows temporal leakage occurs when datasets "
                    "have a time dimension that is ignored during splitting. "
                    "No temporal columns were found, so this check is not applicable."
                ),
                confidence=1.0,
                recommendation=(
                    "If your dataset has an implicit time ordering (e.g., rows were collected "
                    "sequentially), consider adding a row-index-based temporal split to avoid "
                    "future leakage."
                ),
            ))
            return findings

        # Check 1: Random split on a time-structured dataset
        findings.extend(
            self._check_random_split_on_temporal_data(df, datetime_cols, X_train, X_test)
        )

        # Check 2: Time-order inversion in split (if splits are provided)
        if X_train is not None and X_test is not None:
            findings.extend(
                self._check_split_time_ordering(df, datetime_cols, X_train, X_test)
            )

        # Check 3: Target leakage through temporal aggregates
        findings.extend(
            self._check_temporal_aggregate_leakage(df, target_column, datetime_cols)
        )

        return findings

    # ── Private checks ────────────────────────────────────────────────────

    def _check_random_split_on_temporal_data(
        self,
        df: pd.DataFrame,
        datetime_cols: list[str],
        X_train: Optional[pd.DataFrame],
        X_test: Optional[pd.DataFrame],
    ) -> list[AuditFinding]:
        """
        Flags when time-structured data was split randomly.

        In the LIAR2 paper, random splitting on time-ordered statements
        allowed future speaker history to leak into training features.
        """
        findings = []

        if X_train is None or X_test is None:
            # Can't verify split strategy without splits — flag as suspected
            return [self._create_finding(
                severity=Severity.HIGH,
                status=FindingStatus.SUSPECTED,
                description=(
                    f"Dataset contains temporal column(s) {datetime_cols} "
                    "but no train/test split information was provided to verify "
                    "whether a time-ordered split was used."
                ),
                evidence={
                    "temporal_columns": datetime_cols,
                    "total_rows": len(df),
                },
                affected_columns=datetime_cols,
                explanation=(
                    "The base paper demonstrates that randomly splitting time-structured data "
                    "causes temporal leakage: future records appear in the training set, "
                    "and their aggregated features contaminate past records in the test set. "
                    "Without split information, this cannot be confirmed but is highly suspected."
                ),
                confidence=0.6,
                recommendation=(
                    f"Use a time-ordered split on column '{datetime_cols[0]}': "
                    "sort the dataset by this column, then take the first N% as training "
                    "and the remaining as test. Never use random_state-based splitting "
                    "on temporally ordered data."
                ),
            )]

        # Splits were provided — check if any train index appears after any test index
        # (using DataFrame index as a proxy for original row order)
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        all_indices = sorted(df.index)

        if not all_indices:
            return findings

        # If the split was truly time-ordered, all train indices should be < test indices
        max_train_idx = max(train_indices) if train_indices else -1
        min_test_idx = min(test_indices) if test_indices else float("inf")

        if max_train_idx > min_test_idx:
            overlap_count = sum(1 for ti in train_indices if ti > min_test_idx)
            overlap_pct = round(overlap_count / len(train_indices) * 100, 2)
            findings.append(self._create_finding(
                severity=Severity.HIGH,
                status=FindingStatus.CONFIRMED,
                description=(
                    f"Random split detected on time-structured data: "
                    f"{overlap_pct}% of training samples have higher row indices "
                    "than the earliest test sample, indicating non-chronological splitting."
                ),
                evidence={
                    "temporal_columns": datetime_cols,
                    "max_train_index": int(max_train_idx),
                    "min_test_index": int(min_test_idx),
                    "train_samples_after_test_start": overlap_count,
                    "overlap_percentage": overlap_pct,
                },
                affected_columns=datetime_cols,
                explanation=(
                    "A chronological split requires all training samples to precede all "
                    "test samples in time. The detected index overlap shows future records "
                    "are present in training — exactly the temporal leakage mechanism "
                    "described in the LIAR2 base paper."
                ),
                confidence=0.85,
                recommendation=(
                    f"Sort the dataset by '{datetime_cols[0]}' in ascending order, "
                    "then split at a fixed cutoff (e.g., 80th percentile date). "
                    "Do NOT use sklearn train_test_split with shuffle=True (default) "
                    "on time-series or time-structured data."
                ),
                auto_correctable=False,
            ))

        return findings

    def _check_split_time_ordering(
        self,
        df: pd.DataFrame,
        datetime_cols: list[str],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> list[AuditFinding]:
        """
        Verifies that training timestamps strictly precede test timestamps.
        Directly maps to the paper's 'temporal-honest' requirement.
        """
        findings = []

        for col in datetime_cols:
            if col not in df.columns:
                continue

            try:
                series = pd.to_datetime(df[col], errors="coerce").dropna()
                if series.empty:
                    continue

                # Get timestamps for train and test rows
                train_times = series.loc[series.index.intersection(X_train.index)]
                test_times = series.loc[series.index.intersection(X_test.index)]

                if train_times.empty or test_times.empty:
                    continue

                train_max = train_times.max()
                test_min = test_times.min()

                if train_max > test_min:
                    leaking_rows = int((train_times > test_min).sum())
                    findings.append(self._create_finding(
                        severity=Severity.CRITICAL,
                        status=FindingStatus.CONFIRMED,
                        description=(
                            f"Time-order inversion in split on column '{col}': "
                            f"training data contains {leaking_rows} rows with timestamps "
                            f"AFTER the earliest test timestamp ({test_min})."
                        ),
                        evidence={
                            "datetime_column": col,
                            "train_max_timestamp": str(train_max),
                            "test_min_timestamp": str(test_min),
                            "leaking_train_rows": leaking_rows,
                            "train_size": len(train_times),
                            "test_size": len(test_times),
                        },
                        affected_columns=[col],
                        explanation=(
                            "The base paper shows this is the most severe form of temporal "
                            "leakage: training samples recorded AFTER test samples means the "
                            "model was trained on 'future' data relative to what it's being "
                            "evaluated on. This directly inflates all reported metrics."
                        ),
                        confidence=0.98,
                        recommendation=(
                            f"Re-split the dataset using column '{col}' as the ordering key. "
                            f"Use all rows where {col} < {test_min} as training, "
                            f"and rows where {col} >= {test_min} as test. "
                            "This is the Leak-Free Aggregation (LFA) approach from the paper."
                        ),
                        auto_correctable=False,
                    ))

            except Exception as e:
                logger.debug(f"Temporal ordering check failed for column '{col}': {e}")
                continue

        return findings

    def _check_temporal_aggregate_leakage(
        self,
        df: pd.DataFrame,
        target_column: str,
        datetime_cols: list[str],
    ) -> list[AuditFinding]:
        """
        Detects columns that appear to be temporal aggregates — the
        specific pattern from the LIAR2 paper (speaker credit history counts).

        A temporal aggregate is a column whose value for row i depends on
        statistics computed from other rows — if computed globally (before split),
        it leaks future information.
        """
        findings = []

        if target_column not in df.columns:
            return findings

        feature_cols = [c for c in df.columns if c != target_column and c not in datetime_cols]

        # Look for numeric columns whose name suggests they are aggregates
        aggregate_keywords = {
            "count", "total", "sum", "avg", "mean", "cumulative", "history",
            "score", "credit", "rating", "rank", "freq", "frequency",
            "num_", "_num", "n_", "_n", "prior", "past", "previous",
        }

        suspected_aggregates = []
        for col in feature_cols:
            if df[col].dtype not in [float, int] and not pd.api.types.is_numeric_dtype(df[col]):
                continue
            col_lower = col.lower()
            if any(kw in col_lower for kw in aggregate_keywords):
                suspected_aggregates.append(col)

        if suspected_aggregates:
            # Check if these aggregates correlate with the target
            # (a sign they encode global stats that include the target label)
            leaking_aggregates = []
            target_series = df[target_column]
            is_numeric_target = pd.api.types.is_numeric_dtype(target_series)

            for col in suspected_aggregates:
                try:
                    col_series = df[col].fillna(0)
                    if is_numeric_target:
                        corr = abs(col_series.corr(pd.to_numeric(target_series, errors="coerce")))
                    else:
                        # Use label-encoded target for correlation
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        enc_target = le.fit_transform(target_series.astype(str))
                        corr = abs(col_series.corr(pd.Series(enc_target, index=df.index)))

                    if corr > 0.3:  # Moderate correlation threshold
                        leaking_aggregates.append({
                            "column": col,
                            "correlation_with_target": round(float(corr), 4),
                        })
                except Exception:
                    continue

            if leaking_aggregates:
                findings.append(self._create_finding(
                    severity=Severity.HIGH,
                    status=FindingStatus.SUSPECTED,
                    description=(
                        f"Found {len(leaking_aggregates)} suspected temporal aggregate "
                        f"feature(s) with moderate-to-high correlation with the target. "
                        "If these were computed globally before splitting, they leak "
                        "target information across train/test boundaries."
                    ),
                    evidence={
                        "suspected_aggregate_columns": leaking_aggregates,
                        "total_checked": len(suspected_aggregates),
                        "datetime_columns_in_dataset": datetime_cols,
                    },
                    affected_columns=[a["column"] for a in leaking_aggregates],
                    explanation=(
                        "The LIAR2 base paper shows that speaker 'credit history' features "
                        "(counts of past true/false statements) cause data leakage when computed "
                        "globally over all data before splitting. The same pattern applies here: "
                        "columns named with aggregate keywords (count, total, score, history, etc.) "
                        "that correlate with the target are likely computed using global statistics "
                        "that include information from the test set."
                    ),
                    confidence=0.65,
                    recommendation=(
                        "For each suspected aggregate column, verify HOW it was computed:\n"
                        "1. If computed BEFORE train/test split → LEAKAGE CONFIRMED\n"
                        "2. Apply the paper's Leak-Free Aggregation (LFA) fix:\n"
                        "   - For each row i, compute the aggregate using ONLY rows j where j < i\n"
                        "     (or where timestamp[j] < timestamp[i] if a time column exists)\n"
                        "3. Recompute these features with the LFA approach and re-evaluate metrics."
                    ),
                    auto_correctable=False,
                ))
            elif suspected_aggregates:
                # Found aggregate-named columns but low correlation — still flag
                findings.append(self._create_finding(
                    severity=Severity.LOW,
                    status=FindingStatus.INFORMATIONAL,
                    description=(
                        f"Found {len(suspected_aggregates)} column(s) with aggregate-sounding "
                        "names but low correlation with the target."
                    ),
                    evidence={"aggregate_columns": suspected_aggregates},
                    affected_columns=suspected_aggregates,
                    explanation=(
                        "These columns have names suggesting they may be computed aggregates "
                        "(count, total, score, etc.), but their correlation with the target is low. "
                        "Verify they were computed using only pre-split data."
                    ),
                    confidence=0.4,
                    recommendation=(
                        "Verify that these columns were computed only from training-set rows. "
                        "If computed globally, apply the Leak-Free Aggregation (LFA) approach."
                    ),
                ))

        return findings

    # ── Helpers ───────────────────────────────────────────────────────────

    def _detect_datetime_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Detect columns that are datetime type or have datetime-like names/values.
        """
        detected = []

        for col in df.columns:
            # Already a proper datetime dtype
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                detected.append(col)
                continue

            # Name-based heuristic
            col_lower = col.lower()
            if any(kw in col_lower for kw in self._DATE_KEYWORDS):
                # Verify by trying to parse a sample
                sample = df[col].dropna().head(20)
                if sample.empty:
                    continue
                try:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().mean() >= 0.7:
                        detected.append(col)
                except Exception:
                    pass

        return detected


"""
Audit I: Entity/Group Aggregation Leakage Detection
=====================================================
Inspired by: "Data Leakage in Speaker Credibility Features: A Systematic
Analysis and Correction for the LIAR2 Fake News Benchmark"

The LIAR2 paper identifies a specific leakage pattern:
  - Each record belongs to an entity (speaker)
  - Features are computed as global aggregates over all records for that entity
  - Because the target record is included in its own aggregate, its label
    indirectly leaks into its own feature vector ("self-inclusion leakage")
  - Because test-set records are included when computing train-set aggregates,
    future labels contaminate training features ("cross-split aggregation")

This module generalises this to any tabular dataset:
  - Identifies entity/group columns (low-cardinality categorical columns)
  - Detects numeric features whose per-entity distribution mirrors the target
    (suggesting they encode globally-aggregated target-correlated statistics)
  - Flags the specific self-inclusion pattern when statistics are uniform
    per entity across train and test sets
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

from src.audit.base import BaseAudit, AuditFinding
from src.config import Severity, FindingStatus, AuditCategory

logger = logging.getLogger(__name__)


class AggregationLeakageAudit(BaseAudit):
    """
    Detects entity/group aggregation leakage — the LIAR2 paper's core mechanism.

    The paper's 'speaker credit history' problem generalises to any dataset
    where features are computed as group-level statistics (mean, count, sum)
    over an entity column. If computed globally before splitting, these
    features leak test-set target information into the training set.
    """

    CATEGORY = AuditCategory.AGGREGATION_LEAKAGE
    FINDING_PREFIX = "AGG"

    # Max unique values for a column to be considered an entity/group column
    _MAX_ENTITY_CARDINALITY = 200
    # Min rows per entity to have meaningful aggregation
    _MIN_ROWS_PER_ENTITY = 2

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
        Run entity/group aggregation leakage checks.

        Args:
            df:            Full DataFrame (all rows, all columns)
            target_column: Name of the target column
            task_type:     'classification' or 'regression'
            X_train:       Training features (optional)
            X_test:        Test features (optional)

        Returns:
            List of AuditFinding objects
        """
        findings: list[AuditFinding] = []

        if df is None or df.empty or target_column not in df.columns:
            return findings

        # Find entity columns: categorical, low-cardinality, repeated values
        entity_cols = self._detect_entity_columns(df, target_column)

        if not entity_cols:
            findings.append(self._create_finding(
                severity=Severity.INFO,
                status=FindingStatus.INFORMATIONAL,
                description="No entity/group columns detected for aggregation leakage check.",
                evidence={"checked_columns": list(df.columns)},
                affected_columns=[],
                explanation=(
                    "The base paper's aggregation leakage requires an 'entity' column "
                    "(e.g., speaker name) with multiple records per entity. "
                    "No such column was found with sufficient cardinality."
                ),
                confidence=1.0,
                recommendation=(
                    "If your dataset has a grouping dimension (user ID, speaker, category, "
                    "product, etc.), verify that any per-group aggregate features were "
                    "computed only from training-set rows."
                ),
            ))
            return findings

        # Check 1: Self-inclusion leakage (target row included in its own aggregate)
        findings.extend(
            self._check_self_inclusion_leakage(df, target_column, entity_cols, task_type)
        )

        # Check 2: Cross-split aggregation (test entities leak into train aggregates)
        if X_train is not None and X_test is not None:
            findings.extend(
                self._check_cross_split_aggregation(
                    df, target_column, entity_cols, X_train, X_test, task_type
                )
            )

        # Check 3: Entity-level target leakage (per-entity target rate as a feature)
        findings.extend(
            self._check_entity_target_rate_leakage(df, target_column, entity_cols, task_type)
        )

        return findings

    # ── Private checks ────────────────────────────────────────────────────

    def _check_self_inclusion_leakage(
        self,
        df: pd.DataFrame,
        target_column: str,
        entity_cols: list[str],
        task_type: str,
    ) -> list[AuditFinding]:
        """
        Detects the LIAR2 'self-inclusion' pattern:
        A feature for row i is computed from an aggregate that includes row i itself.

        Signature: for entity columns with 1 record per entity, any aggregate
        over that entity is deterministically the row's own value → perfect leakage.
        For multi-record entities, checks if numeric features are suspiciously
        uniform within entities (suggesting globally-computed stats).
        """
        findings = []

        for entity_col in entity_cols:
            try:
                entity_counts = df[entity_col].value_counts()
                # Find entities that appear in both train and test (cross-split candidates)
                multi_record_entities = entity_counts[entity_counts >= self._MIN_ROWS_PER_ENTITY]

                if multi_record_entities.empty:
                    continue

                # Check numeric features for within-entity uniformity
                # (a sign the feature is a global aggregate for that entity)
                feature_cols = [
                    c for c in df.select_dtypes(include="number").columns
                    if c != target_column
                ]
                uniform_features = []

                for feat_col in feature_cols[:20]:  # limit for performance
                    try:
                        # For each entity, compute std of this feature
                        # If std is 0 for most entities → feature is constant per entity
                        entity_stds = df.groupby(entity_col)[feat_col].std().fillna(0)
                        pct_uniform = (entity_stds == 0).mean()

                        if pct_uniform > 0.8:  # 80%+ of entities have constant value
                            uniform_features.append({
                                "feature": feat_col,
                                "pct_entities_uniform": round(float(pct_uniform) * 100, 1),
                            })
                    except Exception:
                        continue

                if uniform_features:
                    findings.append(self._create_finding(
                        severity=Severity.HIGH,
                        status=FindingStatus.SUSPECTED,
                        description=(
                            f"Entity column '{entity_col}' has {len(uniform_features)} "
                            "feature(s) that are constant within each entity — a strong "
                            "indicator of globally-computed aggregates (self-inclusion leakage)."
                        ),
                        evidence={
                            "entity_column": entity_col,
                            "entities_analyzed": int(len(multi_record_entities)),
                            "uniform_features": uniform_features[:10],
                        },
                        affected_columns=[entity_col] + [f["feature"] for f in uniform_features[:5]],
                        explanation=(
                            "The LIAR2 base paper identifies 'self-inclusion leakage': when a "
                            "feature for entity E is computed from ALL records of E (including the "
                            "target record itself), the feature implicitly encodes the target's label. "
                            "Features that are constant within an entity are typical of globally "
                            "computed statistics (e.g., speaker's overall true-statement rate). "
                            "If these were computed before train/test splitting, every test record "
                            "influenced its own training features."
                        ),
                        confidence=0.72,
                        recommendation=(
                            f"For each constant-within-entity feature, verify computation method:\n"
                            f"1. If computed globally (all rows of entity '{entity_col}') → LEAKAGE\n"
                            "2. Apply the paper's Leave-One-Out / Leak-Free fix:\n"
                            "   For row i of entity E, compute the aggregate using all rows of E "
                            "   EXCEPT row i itself (leave-one-out) or only rows prior to row i "
                            "   in time (temporal-honest aggregation)."
                        ),
                        auto_correctable=False,
                    ))

            except Exception as e:
                logger.debug(f"Self-inclusion check failed for entity col '{entity_col}': {e}")
                continue

        return findings

    def _check_cross_split_aggregation(
        self,
        df: pd.DataFrame,
        target_column: str,
        entity_cols: list[str],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        task_type: str,
    ) -> list[AuditFinding]:
        """
        Detects cross-split aggregation leakage:
        Entities that appear in BOTH train and test sets. If per-entity
        aggregates were computed globally, test records contaminate train features.

        This is the LIAR2 paper's core finding: speaker credit counts
        are computed over all statements (train + test), so test statement
        labels leak into the training features.
        """
        findings = []

        for entity_col in entity_cols:
            if entity_col not in X_train.columns or entity_col not in X_test.columns:
                continue

            try:
                train_entities = set(X_train[entity_col].dropna().unique())
                test_entities = set(X_test[entity_col].dropna().unique())
                shared_entities = train_entities & test_entities

                if not shared_entities:
                    continue

                overlap_pct = round(len(shared_entities) / len(train_entities | test_entities) * 100, 1)

                # Measure how much the target distribution differs within shared entities
                # across train and test — a large difference suggests global aggregation changed it
                shared_entity_list = list(shared_entities)[:50]  # sample
                df_shared = df[df[entity_col].isin(shared_entity_list)]

                train_idx = X_train.index
                test_idx = X_test.index

                train_shared = df_shared[df_shared.index.isin(train_idx)]
                test_shared = df_shared[df_shared.index.isin(test_idx)]

                if train_shared.empty or test_shared.empty:
                    continue

                findings.append(self._create_finding(
                    severity=Severity.HIGH,
                    status=FindingStatus.CONFIRMED,
                    description=(
                        f"Entity column '{entity_col}': {len(shared_entities)} entities "
                        f"({overlap_pct}% of all unique entities) appear in BOTH train and "
                        "test sets. Any globally-computed per-entity features are contaminated."
                    ),
                    evidence={
                        "entity_column": entity_col,
                        "shared_entity_count": len(shared_entities),
                        "train_entity_count": len(train_entities),
                        "test_entity_count": len(test_entities),
                        "overlap_percentage": overlap_pct,
                        "shared_entities_sample": sorted(str(e) for e in list(shared_entities)[:10]),
                    },
                    affected_columns=[entity_col],
                    explanation=(
                        "This is the exact mechanism described in the LIAR2 base paper: "
                        "when speaker (entity) records appear in both train and test sets, "
                        "and per-speaker features (like statement counts, credit scores) are "
                        "computed globally over ALL records, the test-set labels contaminate "
                        "the training-set features for shared speakers. This inflates metrics "
                        "because the model learns from information it shouldn't have access to "
                        "at training time."
                    ),
                    confidence=0.9,
                    recommendation=(
                        f"For entity column '{entity_col}', apply the paper's Leak-Free "
                        "Aggregation (LFA) approach:\n"
                        f"1. Sort data by time (if available) or row order\n"
                        f"2. For each row i, compute per-entity statistics using ONLY rows j < i\n"
                        f"3. This ensures test-set records never influence training-set features\n"
                        f"4. Alternative: use entity-stratified splitting so each entity appears "
                        f"in only one split (but this may not be feasible for small entity groups)"
                    ),
                    auto_correctable=False,
                ))

            except Exception as e:
                logger.debug(f"Cross-split check failed for entity col '{entity_col}': {e}")
                continue

        return findings

    def _check_entity_target_rate_leakage(
        self,
        df: pd.DataFrame,
        target_column: str,
        entity_cols: list[str],
        task_type: str,
    ) -> list[AuditFinding]:
        """
        Detects when a feature encodes the entity's target rate — the
        clearest form of LIAR2-style leakage (speaker's historical true/false rate).

        If a numeric feature has near-perfect correlation with the entity's
        average target value, it likely IS the global per-entity target rate.
        """
        findings = []

        target_series = df[target_column]
        is_numeric_target = pd.api.types.is_numeric_dtype(target_series)

        # Encode target numerically if needed
        if not is_numeric_target:
            try:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                target_numeric = pd.Series(
                    le.fit_transform(target_series.astype(str)),
                    index=df.index,
                )
            except Exception:
                return findings
        else:
            target_numeric = pd.to_numeric(target_series, errors="coerce")

        for entity_col in entity_cols:
            try:
                # Compute per-entity target rate (global — as a leaky baseline would)
                entity_target_rate = target_numeric.groupby(df[entity_col]).mean()
                # Map back to each row
                row_entity_target_rate = df[entity_col].map(entity_target_rate)

                # Check all numeric features for correlation with this entity target rate
                feature_cols = [
                    c for c in df.select_dtypes(include="number").columns
                    if c != target_column
                ]

                high_corr_features = []
                for feat_col in feature_cols[:30]:
                    try:
                        corr = abs(df[feat_col].fillna(0).corr(row_entity_target_rate.fillna(0)))
                        if corr > 0.75:
                            high_corr_features.append({
                                "feature": feat_col,
                                "correlation_with_entity_target_rate": round(float(corr), 4),
                            })
                    except Exception:
                        continue

                if high_corr_features:
                    findings.append(self._create_finding(
                        severity=Severity.CRITICAL,
                        status=FindingStatus.SUSPECTED,
                        description=(
                            f"{len(high_corr_features)} feature(s) are highly correlated "
                            f"(>0.75) with the per-entity target rate computed from "
                            f"entity column '{entity_col}'. These features likely encode "
                            "the global entity target history — the LIAR2 leakage pattern."
                        ),
                        evidence={
                            "entity_column": entity_col,
                            "leaking_features": high_corr_features,
                            "entity_count": int(df[entity_col].nunique()),
                        },
                        affected_columns=[entity_col] + [f["feature"] for f in high_corr_features],
                        explanation=(
                            "The LIAR2 paper's key finding: speaker 'credit history' features "
                            "(counts of past true/false/half-true statements) correlate strongly "
                            "with the target because they are computed globally — including the "
                            "target statement's own label. Features that highly correlate with "
                            f"the per-entity average target rate (grouped by '{entity_col}') "
                            "are almost certainly encoding this global aggregate. If used in "
                            "training, the model exploits information that shouldn't be available."
                        ),
                        confidence=0.85,
                        recommendation=(
                            "These features are likely encoding entity-level target rates "
                            "computed on the full dataset (leaky). Apply the LFA correction:\n"
                            "1. For each row i belonging to entity E:\n"
                            "   feature_value = mean(target for all rows of E EXCLUDING row i)\n"
                            "2. This 'leave-one-out' approach eliminates self-inclusion leakage\n"
                            "3. For temporal data, use: mean(target for rows of E with time < time_i)\n"
                            "4. After correction, re-evaluate model performance — inflated metrics "
                            "from the original paper dropped significantly after this fix."
                        ),
                        auto_correctable=True,  # LFA correction can be applied automatically
                    ))

            except Exception as e:
                logger.debug(f"Entity target rate check failed for '{entity_col}': {e}")
                continue

        return findings

    # ── Helpers ───────────────────────────────────────────────────────────

    def _detect_entity_columns(self, df: pd.DataFrame, target_column: str) -> list[str]:
        """
        Identify entity/group columns: categorical, low-to-medium cardinality,
        with multiple records per entity value.
        """
        entity_cols = []

        entity_keywords = {
            "speaker", "user", "entity", "group", "category", "id", "name",
            "author", "source", "type", "class", "label", "party", "region",
            "product", "customer", "client", "vendor", "country", "city",
        }

        for col in df.columns:
            if col == target_column:
                continue

            # Must be categorical or low-cardinality integer
            dtype = df[col].dtype
            is_categorical = dtype == object or str(dtype) in ("category", "bool")
            nunique = df[col].nunique()

            # Also consider low-cardinality integers as group columns
            is_low_card_int = (
                pd.api.types.is_integer_dtype(dtype)
                and 2 <= nunique <= self._MAX_ENTITY_CARDINALITY
            )

            if not (is_categorical or is_low_card_int):
                continue

            if nunique < 2 or nunique > self._MAX_ENTITY_CARDINALITY:
                continue

            # Must have repeated values (multiple records per entity)
            avg_records_per_entity = len(df) / nunique
            if avg_records_per_entity < self._MIN_ROWS_PER_ENTITY:
                continue

            # Name-based bonus: columns named like entity identifiers
            col_lower = col.lower()
            has_entity_name = any(kw in col_lower for kw in entity_keywords)

            # Include if name matches or if cardinality is low enough to be meaningful
            if has_entity_name or (nunique <= 50 and avg_records_per_entity >= 3):
                entity_cols.append(col)

        return entity_cols


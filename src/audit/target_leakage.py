"""
Audit A: Target Leakage Detection

Detects features that directly or indirectly encode the target variable:
- Columns identical to the target
- Columns that are direct copies/encodings of the target
- Features with suspiciously perfect correlation to target
- Target-derived identifiers
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from src.audit.base import BaseAudit, AuditFinding
from src.config import (
    Severity, FindingStatus, AuditCategory,
    TARGET_LEAKAGE_CORRELATION_THRESHOLD,
    TARGET_LEAKAGE_MUTUAL_INFO_THRESHOLD,
)

logger = logging.getLogger(__name__)


class TargetLeakageAudit(BaseAudit):
    """Detects direct and indirect target leakage."""

    CATEGORY = AuditCategory.TARGET_LEAKAGE
    FINDING_PREFIX = "TL"

    def run(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str = "classification",
        **kwargs,
    ) -> list[AuditFinding]:
        """
        Run target leakage checks.

        Args:
            df: Full DataFrame including target
            target_column: Name of the target column
            task_type: "classification" or "regression"

        Returns:
            List of AuditFinding objects
        """
        findings = []

        if target_column not in df.columns:
            return findings

        target = df[target_column]
        features = df.drop(columns=[target_column])

        # Check 1: Identical columns
        findings.extend(self._check_identical_columns(features, target))

        # Check 2: Perfect correlation (numeric only)
        findings.extend(self._check_high_correlation(features, target))

        # Check 3: One-to-one mapping (categorical)
        findings.extend(self._check_value_mapping(features, target))

        # Check 4: High mutual information
        findings.extend(
            self._check_mutual_information(features, target, task_type)
        )

        # Check 5: Suspicious column names
        findings.extend(
            self._check_suspicious_names(features, target_column)
        )

        return findings

    def _check_identical_columns(
        self, features: pd.DataFrame, target: pd.Series
    ) -> list[AuditFinding]:
        """Check for columns that are identical to the target."""
        findings = []

        for col in features.columns:
            try:
                series = features[col]

                # Skip if types don't match and can't compare
                if series.dtype != target.dtype:
                    # Try comparing after converting
                    try:
                        if series.astype(str).equals(target.astype(str)):
                            findings.append(self._create_finding(
                                severity=Severity.CRITICAL,
                                status=FindingStatus.CONFIRMED,
                                description=f"Column '{col}' is identical to target (after string conversion).",
                                evidence={
                                    "column": col,
                                    "match_type": "identical_after_conversion",
                                    "target_dtype": str(target.dtype),
                                    "feature_dtype": str(series.dtype),
                                    "sample_values": series.head(5).tolist(),
                                },
                                affected_columns=[col],
                                explanation=(
                                    f"The feature '{col}' contains the exact same values as the target. "
                                    "This is direct data leakage — the model can perfectly predict "
                                    "the target by simply reading this feature."
                                ),
                                confidence=1.0,
                                recommendation=f"Remove column '{col}' from features before training.",
                                auto_correctable=True,
                            ))
                    except (ValueError, TypeError):
                        pass
                    continue

                # Direct comparison
                if series.equals(target):
                    findings.append(self._create_finding(
                        severity=Severity.CRITICAL,
                        status=FindingStatus.CONFIRMED,
                        description=f"Column '{col}' is identical to target column.",
                        evidence={
                            "column": col,
                            "match_type": "exact_identical",
                            "sample_values": series.head(5).tolist(),
                        },
                        affected_columns=[col],
                        explanation=(
                            f"The feature '{col}' is an exact copy of the target variable. "
                            "This represents direct data leakage."
                        ),
                        confidence=1.0,
                        recommendation=f"Remove column '{col}' from features.",
                        auto_correctable=True,
                    ))

            except Exception as e:
                logger.debug(f"Error checking column {col}: {e}")

        return findings

    def _check_high_correlation(
        self, features: pd.DataFrame, target: pd.Series
    ) -> list[AuditFinding]:
        """Check for features with suspiciously high correlation to target."""
        findings = []

        # Only check numeric columns against numeric target
        if not pd.api.types.is_numeric_dtype(target):
            return findings

        numeric_features = features.select_dtypes(include=[np.number])

        for col in numeric_features.columns:
            try:
                series = numeric_features[col].dropna()
                target_aligned = target.loc[series.index].dropna()

                # Align
                common_idx = series.index.intersection(target_aligned.index)
                if len(common_idx) < 10:
                    continue

                s = series.loc[common_idx]
                t = target_aligned.loc[common_idx]

                # Skip if either has zero variance
                if s.std() == 0 or t.std() == 0:
                    continue

                corr = abs(float(s.corr(t)))

                if corr >= TARGET_LEAKAGE_CORRELATION_THRESHOLD:
                    findings.append(self._create_finding(
                        severity=Severity.HIGH,
                        status=FindingStatus.SUSPECTED,
                        description=(
                            f"Column '{col}' has very high correlation "
                            f"({corr:.4f}) with the target."
                        ),
                        evidence={
                            "column": col,
                            "correlation": round(corr, 4),
                            "threshold": TARGET_LEAKAGE_CORRELATION_THRESHOLD,
                            "n_samples": len(common_idx),
                        },
                        affected_columns=[col],
                        explanation=(
                            f"The feature '{col}' has a {corr:.2%} correlation with the target. "
                            "While high correlation alone does not prove leakage, "
                            "correlation above 98% is unusual and warrants investigation. "
                            "Determine whether this feature would be available at prediction time."
                        ),
                        confidence=min(corr, 0.95),
                        recommendation=(
                            f"Investigate whether '{col}' is derived from the target or "
                            "represents information not available at prediction time. "
                            "A high correlation can be legitimate (strong predictor) or "
                            "illegitimate (post-outcome feature)."
                        ),
                        auto_correctable=False,
                    ))

            except Exception as e:
                logger.debug(f"Correlation check failed for {col}: {e}")

        return findings

    def _check_value_mapping(
        self, features: pd.DataFrame, target: pd.Series
    ) -> list[AuditFinding]:
        """Check for columns with a one-to-one mapping to the target."""
        findings = []

        # Focus on categorical/low-cardinality columns
        for col in features.columns:
            series = features[col].dropna()
            target_aligned = target.loc[series.index].dropna()

            common_idx = series.index.intersection(target_aligned.index)
            if len(common_idx) < 10:
                continue

            s = series.loc[common_idx]
            t = target_aligned.loc[common_idx]

            # Check if there's a 1:1 mapping between feature and target values
            try:
                if s.nunique() > 50:
                    continue  # Skip high-cardinality columns

                # Group by feature value, check if target is constant within each group
                groups = pd.DataFrame({"feature": s, "target": t}).groupby("feature")["target"]
                n_unique_per_group = groups.nunique()

                if (n_unique_per_group == 1).all() and s.nunique() == t.nunique():
                    findings.append(self._create_finding(
                        severity=Severity.CRITICAL,
                        status=FindingStatus.CONFIRMED,
                        description=(
                            f"Column '{col}' has a perfect one-to-one mapping "
                            "with the target."
                        ),
                        evidence={
                            "column": col,
                            "mapping": {
                                str(k): str(v)
                                for k, v in zip(
                                    n_unique_per_group.index[:10],
                                    groups.first().values[:10]
                                )
                            },
                            "feature_unique": int(s.nunique()),
                            "target_unique": int(t.nunique()),
                        },
                        affected_columns=[col],
                        explanation=(
                            f"Every unique value of '{col}' maps to exactly one target value, "
                            "and vice versa. This is effectively a lookup table for the target."
                        ),
                        confidence=1.0,
                        recommendation=f"Remove column '{col}' — it encodes the target directly.",
                        auto_correctable=True,
                    ))
                elif (n_unique_per_group == 1).all() and s.nunique() < t.nunique():
                    # Many-to-one: feature perfectly predicts target
                    findings.append(self._create_finding(
                        severity=Severity.HIGH,
                        status=FindingStatus.SUSPECTED,
                        description=(
                            f"Column '{col}' perfectly determines the target value "
                            "(many-to-one mapping)."
                        ),
                        evidence={
                            "column": col,
                            "mapping_type": "many_to_one",
                            "feature_unique": int(s.nunique()),
                            "target_unique": int(t.nunique()),
                        },
                        affected_columns=[col],
                        explanation=(
                            f"Each unique value of '{col}' corresponds to exactly one target value. "
                            "This could indicate the feature was derived from the target."
                        ),
                        confidence=0.85,
                        recommendation=(
                            f"Investigate whether '{col}' is a post-outcome encoding. "
                            "If it would not be available at prediction time, remove it."
                        ),
                        auto_correctable=False,
                    ))

            except Exception as e:
                logger.debug(f"Value mapping check failed for {col}: {e}")

        return findings

    def _check_mutual_information(
        self, features: pd.DataFrame, target: pd.Series,
        task_type: str
    ) -> list[AuditFinding]:
        """Check for suspiciously high mutual information."""
        findings = []

        try:
            # Only use numeric features for MI calculation
            numeric_features = features.select_dtypes(include=[np.number])

            if numeric_features.empty:
                return findings

            # Align and drop NaN
            combined = pd.concat([numeric_features, target.rename("__target__")], axis=1)
            combined = combined.dropna()

            if len(combined) < 20:
                return findings

            X = combined.drop(columns=["__target__"])
            y = combined["__target__"]

            # Encode target if categorical
            if y.dtype == object or y.dtype.name == "category":
                from sklearn.preprocessing import LabelEncoder
                y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)

            # Calculate MI
            if task_type == "classification":
                mi_scores = mutual_info_classif(
                    X, y, random_state=42, n_neighbors=5
                )
            else:
                mi_scores = mutual_info_regression(
                    X, y, random_state=42, n_neighbors=5
                )

            # Normalize MI scores
            if mi_scores.max() > 0:
                mi_normalized = mi_scores / mi_scores.max()

                for col, mi_norm, mi_raw in zip(X.columns, mi_normalized, mi_scores):
                    if mi_norm >= TARGET_LEAKAGE_MUTUAL_INFO_THRESHOLD:
                        # Don't duplicate findings already caught by correlation
                        findings.append(self._create_finding(
                            severity=Severity.MEDIUM,
                            status=FindingStatus.SUSPECTED,
                            description=(
                                f"Column '{col}' has very high mutual information "
                                f"({mi_raw:.4f}, normalized: {mi_norm:.4f}) with the target."
                            ),
                            evidence={
                                "column": col,
                                "mutual_info_raw": round(float(mi_raw), 4),
                                "mutual_info_normalized": round(float(mi_norm), 4),
                                "threshold": TARGET_LEAKAGE_MUTUAL_INFO_THRESHOLD,
                            },
                            affected_columns=[col],
                            explanation=(
                                f"Mutual information measures the dependency between '{col}' "
                                "and the target. A very high normalized MI suggests this feature "
                                "may contain information about the target that should be investigated."
                            ),
                            confidence=float(mi_norm) * 0.8,
                            recommendation=(
                                f"Verify whether '{col}' is a legitimate predictor or "
                                "a post-outcome variable."
                            ),
                            auto_correctable=False,
                        ))

        except Exception as e:
            logger.warning(f"Mutual information check failed: {e}")

        return findings

    def _check_suspicious_names(
        self, features: pd.DataFrame, target_name: str
    ) -> list[AuditFinding]:
        """Check for columns with names suggesting they're related to the target."""
        findings = []

        suspicious_patterns = [
            "label", "class", "target", "outcome", "result",
            "prediction", "predicted", "output", "response",
        ]

        target_lower = target_name.lower()

        for col in features.columns:
            col_lower = col.lower()

            # Check if column name contains target name
            if target_lower in col_lower and col_lower != target_lower:
                findings.append(self._create_finding(
                    severity=Severity.LOW,
                    status=FindingStatus.INFORMATIONAL,
                    description=(
                        f"Column '{col}' has a name similar to the target '{target_name}'."
                    ),
                    evidence={
                        "column": col,
                        "target_name": target_name,
                        "name_similarity": "contains_target_name",
                    },
                    affected_columns=[col],
                    explanation=(
                        f"The feature name '{col}' contains the target name '{target_name}', "
                        "which may indicate it is derived from or related to the target."
                    ),
                    confidence=0.3,
                    recommendation=(
                        f"Manually verify that '{col}' is not derived from the target "
                        "and would be available at prediction time."
                    ),
                    auto_correctable=False,
                ))

        return findings

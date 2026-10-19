"""
Audit E: Performance Anomaly Detection

Detects suspicious model performance patterns:
- Unusually high train-test performance gaps
- Suspiciously perfect or near-perfect scores
- High variance across CV folds
- Features with extreme predictive power
"""

import logging

import numpy as np

from src.audit.base import BaseAudit, AuditFinding
from src.config import (
    Severity, FindingStatus, AuditCategory,
    TRAIN_TEST_GAP_THRESHOLD, SUSPICIOUSLY_HIGH_ACCURACY,
    CV_VARIANCE_THRESHOLD,
)

logger = logging.getLogger(__name__)


class PerformanceAnomalyAudit(BaseAudit):
    """Detects suspicious performance patterns that may indicate leakage."""

    CATEGORY = AuditCategory.PERFORMANCE_ANOMALY
    FINDING_PREFIX = "PA"

    def run(
        self,
        train_metrics: dict = None,
        test_metrics: dict = None,
        cv_metrics: dict = None,
        baseline_metrics: dict = None,
        feature_importances: dict = None,
        task_type: str = "classification",
        **kwargs,
    ) -> list[AuditFinding]:
        findings = []

        train_metrics = train_metrics or {}
        test_metrics = test_metrics or {}
        cv_metrics = cv_metrics or {}
        baseline_metrics = baseline_metrics or {}

        # Check 1: Train vs test gap (overfitting or leakage)
        findings.extend(
            self._check_train_test_gap(train_metrics, test_metrics, task_type)
        )

        # Check 2: Suspiciously high performance
        findings.extend(
            self._check_suspiciously_high(test_metrics, task_type)
        )

        # Check 3: CV fold variance
        findings.extend(self._check_cv_variance(cv_metrics, task_type))

        # Check 4: Performance far above baseline
        findings.extend(
            self._check_baseline_gap(test_metrics, baseline_metrics, task_type)
        )

        # Check 5: Feature importance anomalies
        if feature_importances:
            findings.extend(
                self._check_feature_importances(feature_importances)
            )

        return findings

    def _check_train_test_gap(
        self, train_metrics: dict, test_metrics: dict, task_type: str
    ) -> list[AuditFinding]:
        findings = []

        primary = "accuracy" if task_type == "classification" else "r2"

        train_score = train_metrics.get(primary)
        test_score = test_metrics.get(primary)

        if train_score is None or test_score is None:
            return findings

        gap = abs(train_score - test_score)

        if gap > TRAIN_TEST_GAP_THRESHOLD:
            # Large gap: could be overfitting
            if train_score > test_score:
                findings.append(self._create_finding(
                    severity=Severity.MEDIUM,
                    status=FindingStatus.SUSPECTED,
                    description=(
                        f"Large train-test performance gap: "
                        f"train {primary}={train_score:.4f}, "
                        f"test {primary}={test_score:.4f} "
                        f"(gap={gap:.4f})."
                    ),
                    evidence={
                        "train_score": round(train_score, 4),
                        "test_score": round(test_score, 4),
                        "gap": round(gap, 4),
                        "metric": primary,
                        "threshold": TRAIN_TEST_GAP_THRESHOLD,
                    },
                    explanation=(
                        f"The model performs much better on training data "
                        f"(gap of {gap:.2%}). This suggests overfitting — "
                        "the model memorized training data rather than learning "
                        "generalizable patterns. This could also indicate data leakage "
                        "if the training score is near-perfect."
                    ),
                    confidence=min(gap * 3, 0.9),
                    recommendation=(
                        "Use regularization, reduce model complexity, or increase "
                        "training data. Also check for data leakage."
                    ),
                    auto_correctable=False,
                ))
            else:
                # Test higher than train — very suspicious
                findings.append(self._create_finding(
                    severity=Severity.HIGH,
                    status=FindingStatus.SUSPECTED,
                    description=(
                        f"Test performance HIGHER than training: "
                        f"train {primary}={train_score:.4f}, "
                        f"test {primary}={test_score:.4f}."
                    ),
                    evidence={
                        "train_score": round(train_score, 4),
                        "test_score": round(test_score, 4),
                        "gap": round(gap, 4),
                        "metric": primary,
                    },
                    explanation=(
                        "Test performance exceeding training performance is unusual "
                        "and may indicate data leakage, where the test set is somehow "
                        "easier or has leaked into training."
                    ),
                    confidence=0.7,
                    recommendation="Investigate potential data leakage or test set issues.",
                    auto_correctable=False,
                ))

        return findings

    def _check_suspiciously_high(
        self, test_metrics: dict, task_type: str
    ) -> list[AuditFinding]:
        findings = []

        if task_type == "classification":
            acc = test_metrics.get("accuracy")
            if acc is not None and acc >= SUSPICIOUSLY_HIGH_ACCURACY:
                findings.append(self._create_finding(
                    severity=Severity.HIGH,
                    status=FindingStatus.SUSPECTED,
                    description=(
                        f"Suspiciously high test accuracy: {acc:.4f} "
                        f"(threshold: {SUSPICIOUSLY_HIGH_ACCURACY})."
                    ),
                    evidence={
                        "accuracy": round(acc, 4),
                        "threshold": SUSPICIOUSLY_HIGH_ACCURACY,
                        "all_test_metrics": {
                            k: round(v, 4) for k, v in test_metrics.items()
                            if isinstance(v, (int, float)) and k != "confusion_matrix"
                        },
                    },
                    explanation=(
                        f"An accuracy of {acc:.2%} is unusually high and may "
                        "indicate data leakage. While perfect classification is "
                        "possible on some datasets, it warrants investigation."
                    ),
                    confidence=0.6,
                    recommendation=(
                        "Check for target leakage, data contamination, or "
                        "trivially separable classes."
                    ),
                    auto_correctable=False,
                ))
        else:
            r2 = test_metrics.get("r2")
            if r2 is not None and r2 >= 0.99:
                findings.append(self._create_finding(
                    severity=Severity.HIGH,
                    status=FindingStatus.SUSPECTED,
                    description=f"Suspiciously high R² score: {r2:.4f}.",
                    evidence={
                        "r2": round(r2, 4),
                        "all_test_metrics": {
                            k: round(v, 4) for k, v in test_metrics.items()
                            if isinstance(v, (int, float))
                        },
                    },
                    explanation=(
                        f"An R² of {r2:.4f} is nearly perfect and may "
                        "indicate data leakage or a trivial relationship."
                    ),
                    confidence=0.6,
                    recommendation="Investigate feature-target relationships for leakage.",
                    auto_correctable=False,
                ))

        return findings

    def _check_cv_variance(
        self, cv_metrics: dict, task_type: str
    ) -> list[AuditFinding]:
        findings = []

        metric = "accuracy" if task_type == "classification" else "r2"
        std_key = f"cv_{metric}_std"
        mean_key = f"cv_{metric}_mean"

        std_val = cv_metrics.get(std_key)
        mean_val = cv_metrics.get(mean_key)

        if std_val is not None and mean_val is not None:
            if std_val > CV_VARIANCE_THRESHOLD:
                findings.append(self._create_finding(
                    severity=Severity.MEDIUM,
                    status=FindingStatus.INFORMATIONAL,
                    description=(
                        f"High cross-validation variance: "
                        f"mean={mean_val:.4f}, std={std_val:.4f}."
                    ),
                    evidence={
                        "cv_mean": round(mean_val, 4),
                        "cv_std": round(std_val, 4),
                        "metric": metric,
                        "threshold": CV_VARIANCE_THRESHOLD,
                    },
                    explanation=(
                        f"The CV standard deviation of {std_val:.4f} indicates "
                        "performance varies significantly across folds. This may "
                        "indicate small dataset size, class imbalance, or "
                        "inconsistent data quality."
                    ),
                    confidence=0.5,
                    recommendation=(
                        "Consider using more CV folds, stratified splitting, "
                        "or investigating data quality."
                    ),
                    auto_correctable=False,
                ))

        return findings

    def _check_baseline_gap(
        self, test_metrics: dict, baseline_metrics: dict, task_type: str
    ) -> list[AuditFinding]:
        findings = []

        primary = "accuracy" if task_type == "classification" else "r2"

        model_score = test_metrics.get(primary)
        baseline_score = baseline_metrics.get(primary)

        if model_score is None or baseline_score is None:
            return findings

        improvement = model_score - baseline_score

        if improvement <= 0:
            findings.append(self._create_finding(
                severity=Severity.MEDIUM,
                status=FindingStatus.INFORMATIONAL,
                description=(
                    f"Model does not outperform baseline: "
                    f"model={model_score:.4f}, baseline={baseline_score:.4f}."
                ),
                evidence={
                    "model_score": round(model_score, 4),
                    "baseline_score": round(baseline_score, 4),
                    "improvement": round(improvement, 4),
                    "metric": primary,
                },
                explanation=(
                    "The trained model does not exceed the performance of a "
                    "simple baseline (most frequent class for classification, "
                    "mean prediction for regression). This may indicate "
                    "insufficient feature signal or a modeling issue."
                ),
                confidence=0.8,
                recommendation=(
                    "Review feature engineering, try different models, or "
                    "investigate whether the task is learnable from the features."
                ),
                auto_correctable=False,
            ))

        return findings

    def _check_feature_importances(
        self, feature_importances: dict
    ) -> list[AuditFinding]:
        findings = []

        if not feature_importances:
            return findings

        values = list(feature_importances.values())
        if not values:
            return findings

        total = sum(abs(v) for v in values)
        if total == 0:
            return findings

        # Check if one feature dominates
        max_feature = max(feature_importances, key=lambda k: abs(feature_importances[k]))
        max_importance = abs(feature_importances[max_feature]) / total

        if max_importance > 0.8:
            findings.append(self._create_finding(
                severity=Severity.MEDIUM,
                status=FindingStatus.SUSPECTED,
                description=(
                    f"Feature '{max_feature}' dominates with "
                    f"{max_importance:.1%} of total feature importance."
                ),
                evidence={
                    "dominant_feature": max_feature,
                    "relative_importance": round(max_importance, 4),
                    "top_5_features": dict(
                        sorted(
                            feature_importances.items(),
                            key=lambda x: abs(x[1]), reverse=True
                        )[:5]
                    ),
                },
                affected_columns=[max_feature],
                explanation=(
                    f"The feature '{max_feature}' contributes {max_importance:.1%} "
                    "of all predictive power. While this can be legitimate, "
                    "a single feature dominating this strongly may indicate "
                    "target leakage or a feature that encodes the outcome."
                ),
                confidence=0.5,
                recommendation=(
                    f"Investigate whether '{max_feature}' would be available "
                    "at prediction time and is not derived from the target."
                ),
                auto_correctable=False,
            ))

        return findings

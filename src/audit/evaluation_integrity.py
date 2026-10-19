"""
Audit F: Evaluation Integrity

Checks evaluation methodology for correctness:
- Appropriate metric selection for the task
- Class imbalance awareness
- Baseline comparison
- Test-set misuse
- Missing important metrics
"""

import logging

import pandas as pd
import numpy as np

from src.audit.base import BaseAudit, AuditFinding
from src.config import Severity, FindingStatus, AuditCategory, CLASS_IMBALANCE_THRESHOLD

logger = logging.getLogger(__name__)


class EvaluationIntegrityAudit(BaseAudit):
    """Checks evaluation methodology for correctness."""

    CATEGORY = AuditCategory.EVALUATION_INTEGRITY
    FINDING_PREFIX = "EI"

    def run(
        self,
        test_metrics: dict = None,
        train_metrics: dict = None,
        baseline_metrics: dict = None,
        target_analysis: dict = None,
        task_type: str = "classification",
        split_config: dict = None,
        **kwargs,
    ) -> list[AuditFinding]:
        findings = []

        test_metrics = test_metrics or {}
        target_analysis = target_analysis or {}
        split_config = split_config or {}

        # Check 1: Misleading accuracy on imbalanced data
        if task_type == "classification":
            findings.extend(
                self._check_imbalanced_accuracy(test_metrics, target_analysis)
            )

        # Check 2: Missing important metrics
        findings.extend(
            self._check_missing_metrics(test_metrics, task_type)
        )

        # Check 3: No baseline comparison
        findings.extend(
            self._check_baseline_presence(baseline_metrics)
        )

        # Check 4: Confusion matrix analysis
        if task_type == "classification":
            findings.extend(
                self._check_confusion_matrix(test_metrics)
            )

        return findings

    def _check_imbalanced_accuracy(
        self, test_metrics: dict, target_analysis: dict
    ) -> list[AuditFinding]:
        findings = []

        is_imbalanced = target_analysis.get("is_imbalanced", False)
        class_balance = target_analysis.get("class_balance", {})
        accuracy = test_metrics.get("accuracy")

        if is_imbalanced and accuracy is not None:
            min_class = min(class_balance.values()) if class_balance else 0
            majority_pct = max(class_balance.values()) if class_balance else 50

            findings.append(self._create_finding(
                severity=Severity.MEDIUM,
                status=FindingStatus.INFORMATIONAL,
                description=(
                    f"Accuracy ({accuracy:.4f}) may be misleading — "
                    f"dataset is imbalanced (minority class: {min_class:.1f}%)."
                ),
                evidence={
                    "accuracy": round(accuracy, 4),
                    "class_balance": class_balance,
                    "minority_class_pct": round(min_class, 2),
                    "majority_class_pct": round(majority_pct, 2),
                    "baseline_accuracy": round(majority_pct / 100, 4),
                },
                explanation=(
                    f"With a majority class at {majority_pct:.1f}%, a model that "
                    "always predicts the majority class would achieve "
                    f"{majority_pct:.1f}% accuracy. Accuracy alone is insufficient "
                    "for evaluating performance on imbalanced datasets."
                ),
                confidence=0.8,
                recommendation=(
                    "Use precision, recall, F1-score, PR-AUC, or ROC-AUC "
                    "instead of or in addition to accuracy. Consider "
                    "stratified splitting and class-weighted models."
                ),
                auto_correctable=False,
            ))

        return findings

    def _check_missing_metrics(
        self, test_metrics: dict, task_type: str
    ) -> list[AuditFinding]:
        findings = []

        if task_type == "classification":
            required = ["accuracy", "precision", "recall", "f1_score"]
            recommended = ["roc_auc", "pr_auc"]
        else:
            required = ["mae", "rmse", "r2"]
            recommended = ["median_ae"]

        missing_required = [m for m in required if m not in test_metrics]
        missing_recommended = [m for m in recommended if m not in test_metrics]

        if missing_required:
            findings.append(self._create_finding(
                severity=Severity.LOW,
                status=FindingStatus.INFORMATIONAL,
                description=(
                    f"Missing standard metrics: {', '.join(missing_required)}."
                ),
                evidence={
                    "missing_required": missing_required,
                    "available_metrics": list(test_metrics.keys()),
                },
                explanation=(
                    "A complete evaluation should include multiple metrics "
                    "to provide a balanced view of model performance."
                ),
                confidence=0.9,
                recommendation=f"Add these metrics: {', '.join(missing_required)}.",
                auto_correctable=False,
            ))

        if missing_recommended:
            findings.append(self._create_finding(
                severity=Severity.INFO,
                status=FindingStatus.INFORMATIONAL,
                description=(
                    f"Recommended metrics not reported: {', '.join(missing_recommended)}."
                ),
                evidence={
                    "missing_recommended": missing_recommended,
                },
                explanation="These metrics provide additional evaluation perspective.",
                confidence=0.5,
                recommendation=f"Consider adding: {', '.join(missing_recommended)}.",
                auto_correctable=False,
            ))

        return findings

    def _check_baseline_presence(
        self, baseline_metrics: dict
    ) -> list[AuditFinding]:
        findings = []

        if not baseline_metrics:
            findings.append(self._create_finding(
                severity=Severity.LOW,
                status=FindingStatus.INFORMATIONAL,
                description="No baseline model comparison available.",
                evidence={"baseline_metrics": None},
                explanation=(
                    "Without a baseline comparison, it's difficult to judge "
                    "whether the model provides meaningful predictive power."
                ),
                confidence=0.9,
                recommendation=(
                    "Always compare against a dummy baseline "
                    "(majority class for classification, mean for regression)."
                ),
                auto_correctable=False,
            ))

        return findings

    def _check_confusion_matrix(
        self, test_metrics: dict
    ) -> list[AuditFinding]:
        findings = []

        cm = test_metrics.get("confusion_matrix")
        if cm is None:
            return findings

        cm_array = np.array(cm)

        # Check for zero rows/columns (class never predicted or never present)
        for i in range(len(cm_array)):
            if cm_array[i, :].sum() == 0:
                findings.append(self._create_finding(
                    severity=Severity.MEDIUM,
                    status=FindingStatus.INFORMATIONAL,
                    description=(
                        f"Class {i} has no true instances in the test set."
                    ),
                    evidence={
                        "confusion_matrix": cm,
                        "empty_row": i,
                    },
                    explanation=(
                        f"Class {i} has no instances in the test set, meaning "
                        "the model is not evaluated on this class at all."
                    ),
                    confidence=0.9,
                    recommendation="Ensure all classes are represented in the test set.",
                    auto_correctable=False,
                ))

            if cm_array[:, i].sum() == 0:
                findings.append(self._create_finding(
                    severity=Severity.MEDIUM,
                    status=FindingStatus.INFORMATIONAL,
                    description=(
                        f"Class {i} is never predicted by the model."
                    ),
                    evidence={
                        "confusion_matrix": cm,
                        "never_predicted_class": i,
                    },
                    explanation=(
                        f"The model never predicts class {i}. This often occurs "
                        "with severely imbalanced classes."
                    ),
                    confidence=0.9,
                    recommendation="Use class weighting or resampling techniques.",
                    auto_correctable=False,
                ))

        return findings

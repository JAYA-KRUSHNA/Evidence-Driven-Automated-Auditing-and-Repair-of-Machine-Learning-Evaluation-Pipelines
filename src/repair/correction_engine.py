"""
Correction Engine — Orchestrates corrections for detected issues.

Proposes, applies, and validates corrections:
1. Maps findings to correction rules
2. Generates corrected dataset/config
3. Reruns experiment
4. Compares before/after
"""

import logging
import json
import time
from typing import Optional

import pandas as pd

from src.audit.base import AuditFinding
from src.audit.audit_engine import AuditResult
from src.repair.correction_rules import CorrectionRules, CorrectionAction
from src.experiments.experiment_runner import ExperimentRunner, ExperimentConfig, ExperimentResult
from src.database.models import get_session, init_db
from src.database.repository import CorrectionRepository
from src.config import AuditCategory

logger = logging.getLogger(__name__)


class CorrectionPlan:
    """A plan of corrections to apply."""

    def __init__(self):
        self.actions: list[CorrectionAction] = []
        self.findings_addressed: list[str] = []
        self.warnings: list[str] = []

    @property
    def has_corrections(self) -> bool:
        return len(self.actions) > 0


class CorrectionResult:
    """Result of applying corrections and rerunning."""

    def __init__(self):
        self.plan: Optional[CorrectionPlan] = None
        self.original_experiment: Optional[ExperimentResult] = None
        self.corrected_experiment: Optional[ExperimentResult] = None
        self.original_audit: Optional[AuditResult] = None
        self.corrected_audit: Optional[AuditResult] = None
        self.before_metrics: dict = {}
        self.after_metrics: dict = {}
        self.resolved_findings: list[str] = []
        self.unresolved_findings: list[str] = []
        self.status: str = "pending"
        self.error: Optional[str] = None

    def comparison_summary(self) -> dict:
        return {
            "before_metrics": self.before_metrics,
            "after_metrics": self.after_metrics,
            "resolved_count": len(self.resolved_findings),
            "unresolved_count": len(self.unresolved_findings),
            "corrections_applied": len(self.plan.actions) if self.plan else 0,
        }


class CorrectionEngine:
    """
    Proposes and applies corrections for auto-correctable findings.

    Key principle: Never apply destructive changes without user approval.
    Always preserve the original experiment.
    """

    def __init__(self):
        self.rules = CorrectionRules()

    def propose_corrections(
        self,
        audit_result: AuditResult,
        df: pd.DataFrame,
        target_column: str,
        split_config: dict = None,
    ) -> CorrectionPlan:
        """
        Analyze findings and propose corrections.

        Returns a CorrectionPlan — does NOT apply anything.
        """
        plan = CorrectionPlan()
        split_config = split_config or {}

        for finding in audit_result.findings:
            if not finding.auto_correctable:
                continue

            action = self._map_finding_to_action(
                finding, df, target_column, split_config
            )
            if action:
                action.finding_id = finding.finding_id
                plan.actions.append(action)
                plan.findings_addressed.append(finding.finding_id)

        if not plan.has_corrections:
            plan.warnings.append(
                "No auto-correctable findings detected. Manual review recommended."
            )

        return plan

    def apply_corrections(
        self,
        plan: CorrectionPlan,
        df: pd.DataFrame,
        config: ExperimentConfig,
        original_result: ExperimentResult,
        original_audit: AuditResult,
        progress_callback=None,
    ) -> CorrectionResult:
        """
        Apply approved corrections and rerun the experiment.

        Args:
            plan: Approved correction plan
            df: Original dataset
            config: Original experiment config
            original_result: Original experiment result
            original_audit: Original audit result
            progress_callback: Optional progress callback
        """
        result = CorrectionResult()
        result.plan = plan
        result.original_experiment = original_result
        result.original_audit = original_audit

        try:
            # Step 1: Apply data corrections
            df_corrected = df.copy()
            split_config = config.get_split_config() if hasattr(config, 'get_split_config') else {}

            for action in plan.actions:
                df_corrected, split_config = self._apply_action(
                    action, df_corrected, config.target_column, split_config
                )

            # Collect before metrics from original
            if original_result.training_results:
                first_model = list(original_result.training_results.values())[0]
                result.before_metrics = first_model.test_metrics.copy()

            # Step 2: Create corrected experiment config
            corrected_config = ExperimentConfig(
                dataset_id=config.dataset_id,
                target_column=config.target_column,
                task_type=config.task_type,
                model_names=config.model_names,
                test_size=config.test_size,
                cv_folds=config.cv_folds,
                random_seed=config.random_seed,
                split_strategy=split_config.get("strategy", config.split_strategy),
                experiment_name=f"{config.experiment_name}_corrected",
            )

            # Step 3: Rerun experiment
            runner = ExperimentRunner()
            corrected_result = runner.run(
                df_corrected, corrected_config, progress_callback
            )
            result.corrected_experiment = corrected_result

            # Collect after metrics
            if corrected_result.training_results:
                first_model = list(corrected_result.training_results.values())[0]
                result.after_metrics = first_model.test_metrics.copy()

            # Step 4: Rerun audit on corrected experiment
            from src.audit.audit_engine import AuditEngine

            audit_engine = AuditEngine()

            # Get split data from corrected experiment for audit
            from src.experiments.split_manager import SplitManager
            from src.experiments.pipeline_builder import PipelineBuilder

            X = df_corrected.drop(columns=[config.target_column])
            y = df_corrected[config.target_column]

            if config.task_type == "classification":
                y, _ = PipelineBuilder.encode_target(y, "classification")

            X_train, X_test, y_train, y_test = SplitManager.train_test_split(
                X, y,
                test_size=config.test_size,
                random_seed=config.random_seed,
                task_type=config.task_type,
            )

            corrected_audit = audit_engine.run_full_audit(
                df=df_corrected,
                target_column=config.target_column,
                task_type=config.task_type,
                experiment_id=corrected_result.experiment_id,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                train_metrics=result.after_metrics,
                test_metrics=result.after_metrics,
            )
            result.corrected_audit = corrected_audit

            # Step 5: Determine resolved/unresolved
            original_finding_ids = {f.finding_id for f in original_audit.findings}
            corrected_finding_ids = {f.finding_id for f in corrected_audit.findings}

            # Check by category — did the corrected version have fewer findings in those categories?
            original_by_cat = original_audit.findings_by_category()
            corrected_by_cat = corrected_audit.findings_by_category()

            for finding in original_audit.findings:
                cat = finding.category
                orig_count = len(original_by_cat.get(cat, []))
                corr_count = len(corrected_by_cat.get(cat, []))

                if finding.finding_id in plan.findings_addressed:
                    if corr_count < orig_count:
                        result.resolved_findings.append(finding.finding_id)
                    else:
                        result.unresolved_findings.append(finding.finding_id)

            result.status = "completed"

            # Store correction in DB
            self._store_corrections(result, plan, original_result, corrected_result)

        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            logger.error(f"Correction failed: {e}", exc_info=True)

        return result

    def _map_finding_to_action(
        self,
        finding: AuditFinding,
        df: pd.DataFrame,
        target_column: str,
        split_config: dict,
    ) -> Optional[CorrectionAction]:
        """Map a finding to a correction action."""
        category = finding.category

        if category == AuditCategory.TARGET_LEAKAGE:
            cols = finding.affected_columns
            if cols and finding.status == "confirmed":
                _, action = self.rules.remove_leaky_columns(
                    df, cols, target_column
                )
                return action

        elif category == AuditCategory.SPLIT_CONTAMINATION:
            if "duplicate" in finding.description.lower():
                _, action = self.rules.remove_duplicates(df)
                return action

        elif category == AuditCategory.DATA_QUALITY:
            if "constant" in finding.description.lower():
                _, action = self.rules.remove_constant_columns(df, target_column)
                return action
            elif "infinite" in finding.description.lower():
                _, action = self.rules.replace_infinite_values(df)
                return action
            elif "missing" in finding.description.lower() and "50%" in finding.description:
                _, action = self.rules.remove_high_missing_columns(
                    df, threshold=0.5, target_column=target_column
                )
                return action

        elif category == AuditCategory.SPLIT_STRATEGY:
            evidence = finding.evidence if isinstance(finding.evidence, dict) else {}
            temporal_cols = evidence.get("temporal_columns", [])
            group_col = evidence.get("column", "")

            _, action = self.rules.fix_split_strategy(
                split_config,
                temporal_columns=temporal_cols if temporal_cols else None,
                group_columns=[group_col] if group_col else None,
                is_imbalanced="imbalance" in finding.description.lower(),
            )
            return action

        return None

    def _apply_action(
        self,
        action: CorrectionAction,
        df: pd.DataFrame,
        target_column: str,
        split_config: dict,
    ) -> tuple[pd.DataFrame, dict]:
        """Apply a single correction action."""
        if action.correction_type == "remove_leaky_columns":
            cols = action.changes.get("removed_columns", [])
            df = df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")

        elif action.correction_type == "remove_duplicates":
            df = df.drop_duplicates().reset_index(drop=True)

        elif action.correction_type == "remove_constant_columns":
            cols = action.changes.get("removed_columns", [])
            df = df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")

        elif action.correction_type == "replace_infinite_values":
            import numpy as np
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        elif action.correction_type == "fix_split_strategy":
            split_config.update(action.changes)

        elif action.correction_type == "remove_high_missing_columns":
            cols = list(action.changes.get("removed_columns", {}).keys())
            df = df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")

        return df, split_config

    def _store_corrections(
        self,
        result: CorrectionResult,
        plan: CorrectionPlan,
        original: ExperimentResult,
        corrected: ExperimentResult,
    ):
        """Store correction results in the database."""
        try:
            init_db()
            session = get_session()

            for action in plan.actions:
                CorrectionRepository.create(
                    session,
                    finding_id=action.finding_id,
                    original_experiment_id=original.experiment_id,
                    corrected_experiment_id=corrected.experiment_id,
                    correction_type=action.correction_type,
                    description=action.description,
                    changes_applied=json.dumps(action.changes, default=str),
                    approved=True,
                    successful=result.status == "completed",
                    before_metrics=json.dumps(
                        {k: v for k, v in result.before_metrics.items()
                         if isinstance(v, (int, float))},
                        default=str
                    ),
                    after_metrics=json.dumps(
                        {k: v for k, v in result.after_metrics.items()
                         if isinstance(v, (int, float))},
                        default=str
                    ),
                )

            session.close()
        except Exception as e:
            logger.error(f"Failed to store corrections: {e}")

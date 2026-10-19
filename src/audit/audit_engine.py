"""
Audit Engine — Main orchestrator for all audit checks.

Runs all applicable audit modules against an experiment and
aggregates findings into a single structured report.
"""

import logging
import time
import json
from typing import Optional

import pandas as pd

from src.audit.base import AuditFinding
from src.audit.target_leakage import TargetLeakageAudit
from src.audit.preprocessing_leakage import PreprocessingLeakageAudit
from src.audit.split_integrity import SplitIntegrityAudit
from src.audit.split_strategy import SplitStrategyAudit
from src.audit.performance_anomalies import PerformanceAnomalyAudit
from src.audit.evaluation_integrity import EvaluationIntegrityAudit
from src.audit.data_quality import DataQualityAudit

from src.database.models import get_session, init_db
from src.database.repository import AuditRunRepository, FindingRepository

logger = logging.getLogger(__name__)


class AuditResult:
    """Container for complete audit results."""

    def __init__(self):
        self.audit_run_id: str = ""
        self.experiment_id: str = ""
        self.findings: list[AuditFinding] = []
        self.duration_seconds: float = 0.0
        self.status: str = "pending"
        self.error: Optional[str] = None

    @property
    def total_findings(self) -> int:
        return len(self.findings)

    @property
    def confirmed_count(self) -> int:
        return sum(1 for f in self.findings if f.status == "confirmed")

    @property
    def suspected_count(self) -> int:
        return sum(1 for f in self.findings if f.status == "suspected")

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "critical")

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "high")

    @property
    def correctable_count(self) -> int:
        return sum(1 for f in self.findings if f.auto_correctable)

    def findings_by_severity(self) -> dict:
        result = {}
        for f in self.findings:
            result.setdefault(f.severity, []).append(f)
        return result

    def findings_by_category(self) -> dict:
        result = {}
        for f in self.findings:
            result.setdefault(f.category, []).append(f)
        return result

    def summary(self) -> dict:
        return {
            "total_findings": self.total_findings,
            "confirmed": self.confirmed_count,
            "suspected": self.suspected_count,
            "critical": self.critical_count,
            "high": self.high_count,
            "auto_correctable": self.correctable_count,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
        }


class AuditEngine:
    """
    Main audit orchestrator.

    Runs all applicable audit checks against an experiment and
    stores results in the database.
    """

    def __init__(self):
        self.target_leakage = TargetLeakageAudit()
        self.preprocessing_leakage = PreprocessingLeakageAudit()
        self.split_integrity = SplitIntegrityAudit()
        self.split_strategy = SplitStrategyAudit()
        self.performance_anomalies = PerformanceAnomalyAudit()
        self.evaluation_integrity = EvaluationIntegrityAudit()
        self.data_quality = DataQualityAudit()

    def run_full_audit(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str = "classification",
        experiment_id: str = "",
        X_train: pd.DataFrame = None,
        X_test: pd.DataFrame = None,
        y_train: pd.Series = None,
        y_test: pd.Series = None,
        train_metrics: dict = None,
        test_metrics: dict = None,
        cv_metrics: dict = None,
        baseline_metrics: dict = None,
        feature_importances: dict = None,
        target_analysis: dict = None,
        split_config: dict = None,
        pipeline=None,
        code_text: str = "",
        progress_callback=None,
    ) -> AuditResult:
        """
        Run all applicable audit checks.

        Args:
            df: Full dataset
            target_column: Target column name
            task_type: "classification" or "regression"
            experiment_id: ID of the experiment being audited
            X_train, X_test: Split feature data
            y_train, y_test: Split target data
            train_metrics, test_metrics: Model metrics
            cv_metrics: Cross-validation metrics
            baseline_metrics: Baseline model metrics
            feature_importances: Feature importance dict
            target_analysis: Target column analysis
            split_config: Splitting configuration
            pipeline: sklearn Pipeline object
            code_text: Python source code for static analysis
            progress_callback: Optional callable(step, total, message)
        """
        result = AuditResult()
        result.experiment_id = experiment_id
        start_time = time.time()

        def report(step, total, msg):
            if progress_callback:
                progress_callback(step, total, msg)
            logger.info(f"[Audit {step}/{total}] {msg}")

        total_steps = 7

        try:
            # Create audit run in DB
            result.audit_run_id = self._create_audit_run(experiment_id)

            # Audit A: Target Leakage
            report(1, total_steps, "Checking for target leakage...")
            try:
                findings = self.target_leakage.run(
                    df=df, target_column=target_column, task_type=task_type
                )
                result.findings.extend(findings)
            except Exception as e:
                logger.error(f"Target leakage audit failed: {e}")

            # Audit B: Preprocessing Leakage
            report(2, total_steps, "Checking for preprocessing leakage...")
            try:
                findings = self.preprocessing_leakage.run(
                    pipeline=pipeline,
                    code_text=code_text,
                    X_train=X_train,
                    X_test=X_test,
                )
                result.findings.extend(findings)
            except Exception as e:
                logger.error(f"Preprocessing leakage audit failed: {e}")

            # Audit C: Split Contamination
            report(3, total_steps, "Checking split integrity...")
            try:
                if X_train is not None and X_test is not None:
                    findings = self.split_integrity.run(
                        X_train=X_train, X_test=X_test,
                        y_train=y_train, y_test=y_test,
                        df_full=df,
                    )
                    result.findings.extend(findings)
            except Exception as e:
                logger.error(f"Split integrity audit failed: {e}")

            # Audit D: Split Strategy
            report(4, total_steps, "Validating split strategy...")
            try:
                findings = self.split_strategy.run(
                    df=df, target_column=target_column,
                    task_type=task_type, split_config=split_config or {},
                )
                result.findings.extend(findings)
            except Exception as e:
                logger.error(f"Split strategy audit failed: {e}")

            # Audit E: Performance Anomalies
            report(5, total_steps, "Analyzing performance patterns...")
            try:
                findings = self.performance_anomalies.run(
                    train_metrics=train_metrics or {},
                    test_metrics=test_metrics or {},
                    cv_metrics=cv_metrics or {},
                    baseline_metrics=baseline_metrics or {},
                    feature_importances=feature_importances,
                    task_type=task_type,
                )
                result.findings.extend(findings)
            except Exception as e:
                logger.error(f"Performance anomaly audit failed: {e}")

            # Audit F: Evaluation Integrity
            report(6, total_steps, "Checking evaluation integrity...")
            try:
                findings = self.evaluation_integrity.run(
                    test_metrics=test_metrics or {},
                    train_metrics=train_metrics or {},
                    baseline_metrics=baseline_metrics or {},
                    target_analysis=target_analysis or {},
                    task_type=task_type,
                    split_config=split_config or {},
                )
                result.findings.extend(findings)
            except Exception as e:
                logger.error(f"Evaluation integrity audit failed: {e}")

            # Audit G: Data Quality
            report(7, total_steps, "Checking data quality...")
            try:
                findings = self.data_quality.run(
                    df=df, target_column=target_column, task_type=task_type,
                )
                result.findings.extend(findings)
            except Exception as e:
                logger.error(f"Data quality audit failed: {e}")

            # Store findings
            result.status = "completed"
            result.duration_seconds = round(time.time() - start_time, 2)

            self._store_findings(result)

            logger.info(
                f"Audit complete: {result.total_findings} findings "
                f"({result.confirmed_count} confirmed, "
                f"{result.suspected_count} suspected) "
                f"in {result.duration_seconds}s"
            )

        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            result.duration_seconds = round(time.time() - start_time, 2)
            logger.error(f"Audit failed: {e}", exc_info=True)

        return result

    def _create_audit_run(self, experiment_id: str) -> str:
        """Create an audit run record in the database."""
        try:
            init_db()
            session = get_session()
            audit_run = AuditRunRepository.create(
                session,
                experiment_id=experiment_id,
                audit_type="full",
                status="running",
            )
            session.close()
            return audit_run.id
        except Exception as e:
            logger.error(f"Failed to create audit run: {e}")
            return f"local_audit_{int(time.time())}"

    def _store_findings(self, result: AuditResult):
        """Store audit findings in the database."""
        try:
            session = get_session()

            # Store each finding
            findings_data = [
                f.to_db_dict(result.audit_run_id) for f in result.findings
            ]
            if findings_data:
                FindingRepository.create_bulk(session, findings_data)

            # Update audit run
            AuditRunRepository.update(
                session, result.audit_run_id,
                status=result.status,
                duration_seconds=result.duration_seconds,
                total_findings=result.total_findings,
                confirmed_count=result.confirmed_count,
                suspected_count=result.suspected_count,
            )

            session.close()
        except Exception as e:
            logger.error(f"Failed to store findings: {e}")

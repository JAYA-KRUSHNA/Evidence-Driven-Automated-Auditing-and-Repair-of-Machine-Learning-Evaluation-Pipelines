"""
Database Repository — Data Access Layer

All database operations go through this module.
This is the ONLY layer that needs to change when migrating to Supabase.
"""

import json
from typing import Optional
from datetime import datetime, timezone

from sqlalchemy import func, desc
from sqlalchemy.orm import Session

from src.database.models import (
    get_session, Dataset, Experiment, Metric, AuditRun,
    Finding, Correction, BenchmarkScenario, BenchmarkResult
)


class DatasetRepository:
    """CRUD operations for datasets."""

    @staticmethod
    def create(session: Session, **kwargs) -> Dataset:
        dataset = Dataset(**kwargs)
        session.add(dataset)
        session.commit()
        session.refresh(dataset)
        return dataset

    @staticmethod
    def get_by_id(session: Session, dataset_id: str) -> Optional[Dataset]:
        return session.get(Dataset, dataset_id)

    @staticmethod
    def get_by_source_id(session: Session, source: str, source_id: str) -> Optional[Dataset]:
        return session.query(Dataset).filter(
            Dataset.source == source,
            Dataset.source_id == source_id
        ).first()

    @staticmethod
    def get_all(session: Session) -> list[Dataset]:
        return session.query(Dataset).order_by(desc(Dataset.created_at)).all()

    @staticmethod
    def search(session: Session, query: str = "", source: str = "",
               task_type: str = "") -> list[Dataset]:
        q = session.query(Dataset)
        if query:
            q = q.filter(Dataset.name.ilike(f"%{query}%"))
        if source:
            q = q.filter(Dataset.source == source)
        if task_type:
            q = q.filter(Dataset.task_type == task_type)
        return q.order_by(desc(Dataset.created_at)).all()

    @staticmethod
    def count(session: Session) -> int:
        return session.query(func.count(Dataset.id)).scalar() or 0

    @staticmethod
    def delete(session: Session, dataset_id: str):
        dataset = session.get(Dataset, dataset_id)
        if dataset:
            session.delete(dataset)
            session.commit()

    @staticmethod
    def update(session: Session, dataset_id: str, **kwargs) -> Optional[Dataset]:
        dataset = session.get(Dataset, dataset_id)
        if dataset:
            for key, value in kwargs.items():
                setattr(dataset, key, value)
            dataset.updated_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(dataset)
        return dataset


class ExperimentRepository:
    """CRUD operations for experiments."""

    @staticmethod
    def create(session: Session, **kwargs) -> Experiment:
        experiment = Experiment(**kwargs)
        session.add(experiment)
        session.commit()
        session.refresh(experiment)
        return experiment

    @staticmethod
    def get_by_id(session: Session, experiment_id: str) -> Optional[Experiment]:
        return session.get(Experiment, experiment_id)

    @staticmethod
    def get_all(session: Session, limit: int = 50) -> list[Experiment]:
        return session.query(Experiment).order_by(
            desc(Experiment.created_at)
        ).limit(limit).all()

    @staticmethod
    def get_by_dataset(session: Session, dataset_id: str) -> list[Experiment]:
        return session.query(Experiment).filter(
            Experiment.dataset_id == dataset_id
        ).order_by(desc(Experiment.created_at)).all()

    @staticmethod
    def count(session: Session, status: str = "") -> int:
        q = session.query(func.count(Experiment.id))
        if status:
            q = q.filter(Experiment.status == status)
        return q.scalar() or 0

    @staticmethod
    def update_status(session: Session, experiment_id: str,
                      status: str, **kwargs) -> Optional[Experiment]:
        experiment = session.get(Experiment, experiment_id)
        if experiment:
            experiment.status = status
            for key, value in kwargs.items():
                setattr(experiment, key, value)
            session.commit()
            session.refresh(experiment)
        return experiment

    @staticmethod
    def get_recent(session: Session, limit: int = 10) -> list[Experiment]:
        return session.query(Experiment).order_by(
            desc(Experiment.created_at)
        ).limit(limit).all()


class MetricRepository:
    """CRUD operations for metrics."""

    @staticmethod
    def create_bulk(session: Session, experiment_id: str,
                    metrics: dict, split_type: str = "test",
                    fold_number: int = None):
        for name, value in metrics.items():
            if value is not None:
                metric = Metric(
                    experiment_id=experiment_id,
                    metric_name=name,
                    metric_value=float(value),
                    split_type=split_type,
                    fold_number=fold_number,
                )
                session.add(metric)
        session.commit()

    @staticmethod
    def get_by_experiment(session: Session, experiment_id: str,
                          split_type: str = "") -> list[Metric]:
        q = session.query(Metric).filter(Metric.experiment_id == experiment_id)
        if split_type:
            q = q.filter(Metric.split_type == split_type)
        return q.all()

    @staticmethod
    def get_as_dict(session: Session, experiment_id: str,
                    split_type: str = "test") -> dict:
        metrics = MetricRepository.get_by_experiment(session, experiment_id, split_type)
        return {m.metric_name: m.metric_value for m in metrics}


class AuditRunRepository:
    """CRUD operations for audit runs."""

    @staticmethod
    def create(session: Session, **kwargs) -> AuditRun:
        audit_run = AuditRun(**kwargs)
        session.add(audit_run)
        session.commit()
        session.refresh(audit_run)
        return audit_run

    @staticmethod
    def get_by_id(session: Session, audit_run_id: str) -> Optional[AuditRun]:
        return session.get(AuditRun, audit_run_id)

    @staticmethod
    def get_by_experiment(session: Session, experiment_id: str) -> list[AuditRun]:
        return session.query(AuditRun).filter(
            AuditRun.experiment_id == experiment_id
        ).order_by(desc(AuditRun.created_at)).all()

    @staticmethod
    def count(session: Session) -> int:
        return session.query(func.count(AuditRun.id)).scalar() or 0

    @staticmethod
    def update(session: Session, audit_run_id: str, **kwargs) -> Optional[AuditRun]:
        audit_run = session.get(AuditRun, audit_run_id)
        if audit_run:
            for key, value in kwargs.items():
                setattr(audit_run, key, value)
            session.commit()
            session.refresh(audit_run)
        return audit_run


class FindingRepository:
    """CRUD operations for findings."""

    @staticmethod
    def create(session: Session, **kwargs) -> Finding:
        finding = Finding(**kwargs)
        session.add(finding)
        session.commit()
        session.refresh(finding)
        return finding

    @staticmethod
    def create_bulk(session: Session, findings_data: list[dict]) -> list[Finding]:
        findings = []
        for data in findings_data:
            finding = Finding(**data)
            session.add(finding)
            findings.append(finding)
        session.commit()
        for f in findings:
            session.refresh(f)
        return findings

    @staticmethod
    def get_by_audit_run(session: Session, audit_run_id: str) -> list[Finding]:
        return session.query(Finding).filter(
            Finding.audit_run_id == audit_run_id
        ).order_by(Finding.severity).all()

    @staticmethod
    def get_by_experiment(session: Session, experiment_id: str) -> list[Finding]:
        return session.query(Finding).join(AuditRun).filter(
            AuditRun.experiment_id == experiment_id
        ).order_by(Finding.severity).all()

    @staticmethod
    def count_by_severity(session: Session) -> dict:
        results = session.query(
            Finding.severity, func.count(Finding.id)
        ).group_by(Finding.severity).all()
        return {sev: count for sev, count in results}

    @staticmethod
    def count_by_status(session: Session) -> dict:
        results = session.query(
            Finding.status, func.count(Finding.id)
        ).group_by(Finding.status).all()
        return {status: count for status, count in results}

    @staticmethod
    def count_confirmed(session: Session) -> int:
        return session.query(func.count(Finding.id)).filter(
            Finding.status == "confirmed"
        ).scalar() or 0

    @staticmethod
    def count_suspected(session: Session) -> int:
        return session.query(func.count(Finding.id)).filter(
            Finding.status == "suspected"
        ).scalar() or 0

    @staticmethod
    def update_resolution(session: Session, finding_id: str,
                          resolution_status: str) -> Optional[Finding]:
        finding = session.get(Finding, finding_id)
        if finding:
            finding.resolution_status = resolution_status
            session.commit()
            session.refresh(finding)
        return finding


class CorrectionRepository:
    """CRUD operations for corrections."""

    @staticmethod
    def create(session: Session, **kwargs) -> Correction:
        correction = Correction(**kwargs)
        session.add(correction)
        session.commit()
        session.refresh(correction)
        return correction

    @staticmethod
    def get_by_finding(session: Session, finding_id: str) -> list[Correction]:
        return session.query(Correction).filter(
            Correction.finding_id == finding_id
        ).all()

    @staticmethod
    def get_by_experiment(session: Session, experiment_id: str) -> list[Correction]:
        return session.query(Correction).filter(
            Correction.original_experiment_id == experiment_id
        ).all()

    @staticmethod
    def update(session: Session, correction_id: str, **kwargs) -> Optional[Correction]:
        correction = session.get(Correction, correction_id)
        if correction:
            for key, value in kwargs.items():
                setattr(correction, key, value)
            session.commit()
            session.refresh(correction)
        return correction


class BenchmarkRepository:
    """CRUD operations for benchmark scenarios and results."""

    @staticmethod
    def create_scenario(session: Session, **kwargs) -> BenchmarkScenario:
        scenario = BenchmarkScenario(**kwargs)
        session.add(scenario)
        session.commit()
        session.refresh(scenario)
        return scenario

    @staticmethod
    def get_all_scenarios(session: Session) -> list[BenchmarkScenario]:
        return session.query(BenchmarkScenario).order_by(
            BenchmarkScenario.created_at
        ).all()

    @staticmethod
    def create_result(session: Session, **kwargs) -> BenchmarkResult:
        result = BenchmarkResult(**kwargs)
        session.add(result)
        session.commit()
        session.refresh(result)
        return result

    @staticmethod
    def get_results_by_scenario(session: Session,
                                 scenario_id: str) -> list[BenchmarkResult]:
        return session.query(BenchmarkResult).filter(
            BenchmarkResult.scenario_id == scenario_id
        ).all()

    @staticmethod
    def get_all_results(session: Session) -> list[BenchmarkResult]:
        return session.query(BenchmarkResult).all()

"""
Database Models — SQLAlchemy ORM

Defines all database tables. Uses SQLAlchemy 2.0 style with mapped_column.
Designed to work with SQLite now and migrate to PostgreSQL/Supabase later.
"""

import uuid
import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    create_engine, String, Integer, Float, Text, Boolean,
    DateTime, ForeignKey, JSON, Index
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship,
    Session, sessionmaker
)

from src.config import DATABASE_URL


# ── Base ───────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


def generate_uuid() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Dataset Model ─────────────────────────────────────────────────────────────

class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[str] = mapped_column(String(100), nullable=False)  # "upload", "openml", "uci", "sklearn"
    source_url: Mapped[Optional[str]] = mapped_column(String(500))
    source_id: Mapped[Optional[str]] = mapped_column(String(100))  # OpenML ID, UCI ID, etc.
    task_type: Mapped[Optional[str]] = mapped_column(String(50))  # "classification", "regression"
    target_column: Mapped[Optional[str]] = mapped_column(String(255))
    row_count: Mapped[Optional[int]] = mapped_column(Integer)
    col_count: Mapped[Optional[int]] = mapped_column(Integer)
    missing_pct: Mapped[Optional[float]] = mapped_column(Float)
    column_info: Mapped[Optional[str]] = mapped_column(Text)  # JSON string
    file_path: Mapped[Optional[str]] = mapped_column(String(500))
    checksum: Mapped[Optional[str]] = mapped_column(String(64))
    license_info: Mapped[Optional[str]] = mapped_column(String(500))
    description: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    # Relationships
    experiments: Mapped[list["Experiment"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")

    def get_column_info(self) -> dict:
        if self.column_info:
            return json.loads(self.column_info)
        return {}

    def set_column_info(self, info: dict):
        self.column_info = json.dumps(info)

    def __repr__(self):
        return f"<Dataset(name='{self.name}', source='{self.source}', rows={self.row_count})>"


# ── Experiment Model ──────────────────────────────────────────────────────────

class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    task_type: Mapped[str] = mapped_column(String(50), nullable=False)
    target_column: Mapped[str] = mapped_column(String(255), nullable=False)
    model_config: Mapped[Optional[str]] = mapped_column(Text)  # JSON
    split_config: Mapped[Optional[str]] = mapped_column(Text)  # JSON
    preprocessing_config: Mapped[Optional[str]] = mapped_column(Text)  # JSON
    status: Mapped[str] = mapped_column(String(50), default="pending")  # pending, running, completed, failed
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float)
    random_seed: Mapped[int] = mapped_column(Integer, default=42)
    software_versions: Mapped[Optional[str]] = mapped_column(Text)  # JSON
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    is_corrected: Mapped[bool] = mapped_column(Boolean, default=False)
    original_experiment_id: Mapped[Optional[str]] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    # Relationships
    dataset: Mapped["Dataset"] = relationship(back_populates="experiments")
    metrics: Mapped[list["Metric"]] = relationship(back_populates="experiment", cascade="all, delete-orphan")
    audit_runs: Mapped[list["AuditRun"]] = relationship(back_populates="experiment", cascade="all, delete-orphan")

    def get_model_config(self) -> dict:
        return json.loads(self.model_config) if self.model_config else {}

    def set_model_config(self, config: dict):
        self.model_config = json.dumps(config)

    def get_split_config(self) -> dict:
        return json.loads(self.split_config) if self.split_config else {}

    def set_split_config(self, config: dict):
        self.split_config = json.dumps(config)

    def get_preprocessing_config(self) -> dict:
        return json.loads(self.preprocessing_config) if self.preprocessing_config else {}

    def set_preprocessing_config(self, config: dict):
        self.preprocessing_config = json.dumps(config)

    def __repr__(self):
        return f"<Experiment(name='{self.name}', status='{self.status}')>"


# ── Metric Model ──────────────────────────────────────────────────────────────

class Metric(Base):
    __tablename__ = "metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    experiment_id: Mapped[str] = mapped_column(String(36), ForeignKey("experiments.id"), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    split_type: Mapped[str] = mapped_column(String(50), default="test")  # train, test, cv_mean, cv_fold_N
    fold_number: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    # Relationships
    experiment: Mapped["Experiment"] = relationship(back_populates="metrics")

    def __repr__(self):
        return f"<Metric({self.metric_name}={self.metric_value:.4f}, split={self.split_type})>"


# ── Audit Run Model ───────────────────────────────────────────────────────────

class AuditRun(Base):
    __tablename__ = "audit_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    experiment_id: Mapped[str] = mapped_column(String(36), ForeignKey("experiments.id"), nullable=False)
    audit_type: Mapped[str] = mapped_column(String(100), default="full")
    status: Mapped[str] = mapped_column(String(50), default="pending")
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float)
    total_findings: Mapped[int] = mapped_column(Integer, default=0)
    confirmed_count: Mapped[int] = mapped_column(Integer, default=0)
    suspected_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    # Relationships
    experiment: Mapped["Experiment"] = relationship(back_populates="audit_runs")
    findings: Mapped[list["Finding"]] = relationship(back_populates="audit_run", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<AuditRun(type='{self.audit_type}', findings={self.total_findings})>"


# ── Finding Model ─────────────────────────────────────────────────────────────

class Finding(Base):
    __tablename__ = "findings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    audit_run_id: Mapped[str] = mapped_column(String(36), ForeignKey("audit_runs.id"), nullable=False)
    finding_id: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g. "TL-001"
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    severity: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)  # confirmed, suspected, etc.
    description: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[Optional[str]] = mapped_column(Text)  # JSON
    affected_columns: Mapped[Optional[str]] = mapped_column(Text)  # JSON list
    explanation: Mapped[Optional[str]] = mapped_column(Text)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    recommendation: Mapped[Optional[str]] = mapped_column(Text)
    auto_correctable: Mapped[bool] = mapped_column(Boolean, default=False)
    resolution_status: Mapped[str] = mapped_column(String(50), default="open")  # open, resolved, ignored
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    # Relationships
    audit_run: Mapped["AuditRun"] = relationship(back_populates="findings")
    corrections: Mapped[list["Correction"]] = relationship(back_populates="finding", cascade="all, delete-orphan")

    def get_evidence(self) -> dict:
        return json.loads(self.evidence) if self.evidence else {}

    def set_evidence(self, evidence: dict):
        self.evidence = json.dumps(evidence, default=str)

    def get_affected_columns(self) -> list:
        return json.loads(self.affected_columns) if self.affected_columns else []

    def set_affected_columns(self, columns: list):
        self.affected_columns = json.dumps(columns)

    def __repr__(self):
        return f"<Finding({self.finding_id}: {self.severity} - {self.category})>"


# ── Correction Model ──────────────────────────────────────────────────────────

class Correction(Base):
    __tablename__ = "corrections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    finding_id: Mapped[str] = mapped_column(String(36), ForeignKey("findings.id"), nullable=False)
    original_experiment_id: Mapped[str] = mapped_column(String(36), nullable=False)
    corrected_experiment_id: Mapped[Optional[str]] = mapped_column(String(36))
    correction_type: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    changes_applied: Mapped[Optional[str]] = mapped_column(Text)  # JSON
    approved: Mapped[bool] = mapped_column(Boolean, default=False)
    successful: Mapped[Optional[bool]] = mapped_column(Boolean)
    before_metrics: Mapped[Optional[str]] = mapped_column(Text)  # JSON
    after_metrics: Mapped[Optional[str]] = mapped_column(Text)  # JSON
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    # Relationships
    finding: Mapped["Finding"] = relationship(back_populates="corrections")

    def __repr__(self):
        return f"<Correction(type='{self.correction_type}', approved={self.approved})>"


# ── Benchmark Scenario Model ─────────────────────────────────────────────────

class BenchmarkScenario(Base):
    __tablename__ = "benchmark_scenarios"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    base_dataset_id: Mapped[str] = mapped_column(String(36), nullable=False)
    scenario_name: Mapped[str] = mapped_column(String(255), nullable=False)
    corruption_type: Mapped[str] = mapped_column(String(100), nullable=False)
    corruption_config: Mapped[Optional[str]] = mapped_column(Text)  # JSON
    ground_truth_issues: Mapped[Optional[str]] = mapped_column(Text)  # JSON
    random_seed: Mapped[int] = mapped_column(Integer, default=42)
    is_holdout: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    # Relationships
    benchmark_results: Mapped[list["BenchmarkResult"]] = relationship(
        back_populates="scenario", cascade="all, delete-orphan"
    )

    def get_ground_truth(self) -> list:
        return json.loads(self.ground_truth_issues) if self.ground_truth_issues else []

    def __repr__(self):
        return f"<BenchmarkScenario(name='{self.scenario_name}', type='{self.corruption_type}')>"


# ── Benchmark Result Model ────────────────────────────────────────────────────

class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    scenario_id: Mapped[str] = mapped_column(String(36), ForeignKey("benchmark_scenarios.id"), nullable=False)
    audit_run_id: Mapped[Optional[str]] = mapped_column(String(36))
    detected_issues: Mapped[Optional[str]] = mapped_column(Text)  # JSON
    ground_truth_issues: Mapped[Optional[str]] = mapped_column(Text)  # JSON
    precision_score: Mapped[Optional[float]] = mapped_column(Float)
    recall_score: Mapped[Optional[float]] = mapped_column(Float)
    f1_score: Mapped[Optional[float]] = mapped_column(Float)
    false_positive_rate: Mapped[Optional[float]] = mapped_column(Float)
    per_category_metrics: Mapped[Optional[str]] = mapped_column(Text)  # JSON
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    # Relationships
    scenario: Mapped["BenchmarkScenario"] = relationship(back_populates="benchmark_results")

    def __repr__(self):
        return f"<BenchmarkResult(f1={self.f1_score:.3f})>"


# ── Indexes ────────────────────────────────────────────────────────────────────

Index("idx_datasets_source", Dataset.source)
Index("idx_experiments_dataset", Experiment.dataset_id)
Index("idx_experiments_status", Experiment.status)
Index("idx_audit_runs_experiment", AuditRun.experiment_id)
Index("idx_findings_audit_run", Finding.audit_run_id)
Index("idx_findings_category", Finding.category)
Index("idx_findings_severity", Finding.severity)
Index("idx_metrics_experiment", Metric.experiment_id)


# ── Engine & Session Factory ──────────────────────────────────────────────────

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db():
    """Create all tables. Safe to call multiple times."""
    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    """Get a new database session."""
    return SessionLocal()

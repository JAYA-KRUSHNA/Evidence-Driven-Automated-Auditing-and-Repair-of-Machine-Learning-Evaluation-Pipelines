"""
SQLAlchemy ORM models — 13 tables.

Tables
------
datasets              – uploaded / sample dataset metadata
dataset_versions      – version history per dataset
validation_reports    – overall validation result per dataset
validation_checks     – individual check results
preprocessing_history – applied preprocessing operations
eda_results           – EDA analysis outputs (JSON blob)
experiments           – experiment configuration & status
trained_models        – model metadata, hyperparams, metrics
model_files           – physical file paths for serialised models
predictions           – prediction log
evaluation_results    – per-model evaluation metrics
comparison_runs       – multi-model comparison sessions
reports               – generated report metadata
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Integer, Float, Text, DateTime, Boolean,
    ForeignKey, JSON, Index,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def _uuid() -> str:
    """Generate a short unique identifier."""
    return uuid.uuid4().hex[:12]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. datasets
# ═══════════════════════════════════════════════════════════════════════════════

class Dataset(Base):
    __tablename__ = "datasets"

    id              = Column(String(12), primary_key=True, default=_uuid)
    name            = Column(String(255), nullable=False)
    original_name   = Column(String(255), nullable=False)      # original file name
    file_path       = Column(Text, nullable=False)             # path on disk
    file_format     = Column(String(10), nullable=False)       # csv, xlsx, xls
    source          = Column(String(50), default="upload")     # upload | sample
    num_rows        = Column(Integer, default=0)
    num_columns     = Column(Integer, default=0)
    column_names    = Column(JSON)                             # list[str]
    column_types    = Column(JSON)                             # dict[col→dtype]
    numerical_cols  = Column(JSON)                             # list[str]
    categorical_cols = Column(JSON)                            # list[str]
    datetime_cols   = Column(JSON)                             # list[str]
    identifier_cols = Column(JSON)                             # list[str]
    file_size_bytes = Column(Integer, default=0)
    validation_status = Column(String(30), default="pending")  # pending | valid | valid_with_warnings | invalid
    is_preprocessed = Column(Boolean, default=False)
    upload_time     = Column(DateTime, default=_utcnow)
    updated_at      = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    notes           = Column(Text, default="")

    # relationships
    versions      = relationship("DatasetVersion",      back_populates="dataset", cascade="all, delete-orphan")
    validations   = relationship("ValidationReport",     back_populates="dataset", cascade="all, delete-orphan")
    preprocessings = relationship("PreprocessingHistory", back_populates="dataset", cascade="all, delete-orphan")
    eda_results   = relationship("EDAResult",            back_populates="dataset", cascade="all, delete-orphan")
    experiments   = relationship("Experiment",           back_populates="dataset", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_datasets_name", "name"),
        Index("ix_datasets_source", "source"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  2. dataset_versions
# ═══════════════════════════════════════════════════════════════════════════════

class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id          = Column(String(12), primary_key=True, default=_uuid)
    dataset_id  = Column(String(12), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    version     = Column(Integer, nullable=False, default=1)
    file_path   = Column(Text, nullable=False)
    num_rows    = Column(Integer, default=0)
    num_columns = Column(Integer, default=0)
    description = Column(Text, default="")
    created_at  = Column(DateTime, default=_utcnow)

    dataset = relationship("Dataset", back_populates="versions")


# ═══════════════════════════════════════════════════════════════════════════════
#  3. validation_reports
# ═══════════════════════════════════════════════════════════════════════════════

class ValidationReport(Base):
    __tablename__ = "validation_reports"

    id               = Column(String(12), primary_key=True, default=_uuid)
    dataset_id       = Column(String(12), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    status           = Column(String(30), nullable=False)      # valid | valid_with_warnings | invalid
    total_checks     = Column(Integer, default=0)
    passed_checks    = Column(Integer, default=0)
    failed_checks    = Column(Integer, default=0)
    warning_count    = Column(Integer, default=0)
    is_trainable     = Column(Boolean, default=False)
    summary          = Column(JSON)                            # structured summary dict
    created_at       = Column(DateTime, default=_utcnow)

    dataset = relationship("Dataset", back_populates="validations")
    checks  = relationship("ValidationCheck", back_populates="report", cascade="all, delete-orphan")


# ═══════════════════════════════════════════════════════════════════════════════
#  4. validation_checks
# ═══════════════════════════════════════════════════════════════════════════════

class ValidationCheck(Base):
    __tablename__ = "validation_checks"

    id              = Column(String(12), primary_key=True, default=_uuid)
    report_id       = Column(String(12), ForeignKey("validation_reports.id", ondelete="CASCADE"), nullable=False)
    category        = Column(String(30), nullable=False)       # file | structural | quality | ml_readiness
    check_name      = Column(String(100), nullable=False)
    status          = Column(String(15), nullable=False)       # passed | failed | warning
    message         = Column(Text, default="")
    affected_columns = Column(JSON)                            # list[str] or null
    affected_rows   = Column(Integer, default=0)
    suggestion      = Column(Text, default="")
    details         = Column(JSON)                             # extra data

    report = relationship("ValidationReport", back_populates="checks")


# ═══════════════════════════════════════════════════════════════════════════════
#  5. preprocessing_history
# ═══════════════════════════════════════════════════════════════════════════════

class PreprocessingHistory(Base):
    __tablename__ = "preprocessing_history"

    id             = Column(String(12), primary_key=True, default=_uuid)
    dataset_id     = Column(String(12), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    operation      = Column(String(100), nullable=False)       # e.g. "impute_mean"
    category       = Column(String(50), nullable=False)        # missing_values | encoding | scaling …
    parameters     = Column(JSON)                              # operation-specific config
    affected_columns = Column(JSON)
    rows_before    = Column(Integer, default=0)
    rows_after     = Column(Integer, default=0)
    applied_at     = Column(DateTime, default=_utcnow)

    dataset = relationship("Dataset", back_populates="preprocessings")


# ═══════════════════════════════════════════════════════════════════════════════
#  6. eda_results
# ═══════════════════════════════════════════════════════════════════════════════

class EDAResult(Base):
    __tablename__ = "eda_results"

    id          = Column(String(12), primary_key=True, default=_uuid)
    dataset_id  = Column(String(12), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    analysis_type = Column(String(50), nullable=False)         # overview | statistics | correlation …
    result_data = Column(JSON, nullable=False)                 # the analysis payload
    created_at  = Column(DateTime, default=_utcnow)

    dataset = relationship("Dataset", back_populates="eda_results")


# ═══════════════════════════════════════════════════════════════════════════════
#  7. experiments
# ═══════════════════════════════════════════════════════════════════════════════

class Experiment(Base):
    __tablename__ = "experiments"

    id                   = Column(String(12), primary_key=True, default=_uuid)
    dataset_id           = Column(String(12), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    name                 = Column(String(255), default="")
    task_type            = Column(String(30), nullable=False)  # classification | regression
    target_column        = Column(String(255), nullable=False)
    feature_columns      = Column(JSON)                        # list[str]
    preprocessing_config = Column(JSON)                        # dict
    algorithm            = Column(String(50), nullable=False)
    hyperparameters      = Column(JSON)
    test_size            = Column(Float, default=0.2)
    cv_folds             = Column(Integer, default=5)
    random_seed          = Column(Integer, default=42)
    training_time_sec    = Column(Float, default=0.0)
    status               = Column(String(20), default="pending")  # pending | running | completed | failed
    error_message        = Column(Text, default="")
    created_at           = Column(DateTime, default=_utcnow)
    completed_at         = Column(DateTime, nullable=True)

    dataset      = relationship("Dataset", back_populates="experiments")
    models       = relationship("TrainedModel", back_populates="experiment", cascade="all, delete-orphan")
    evaluations  = relationship("EvaluationResult", back_populates="experiment", cascade="all, delete-orphan")
    predictions  = relationship("Prediction", back_populates="experiment", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_experiments_dataset", "dataset_id"),
        Index("ix_experiments_status", "status"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  8. trained_models
# ═══════════════════════════════════════════════════════════════════════════════

class TrainedModel(Base):
    __tablename__ = "trained_models"

    id              = Column(String(12), primary_key=True, default=_uuid)
    experiment_id   = Column(String(12), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    model_name      = Column(String(100), nullable=False)
    algorithm       = Column(String(50), nullable=False)
    task_type       = Column(String(30), nullable=False)
    hyperparameters = Column(JSON)
    preprocessing_pipeline_path = Column(Text, default="")     # joblib path for pipeline
    metrics         = Column(JSON)                             # evaluation metrics dict
    training_time_sec = Column(Float, default=0.0)
    version         = Column(Integer, default=1)
    is_best         = Column(Boolean, default=False)
    created_at      = Column(DateTime, default=_utcnow)

    experiment  = relationship("Experiment", back_populates="models")
    model_files = relationship("ModelFile", back_populates="model", cascade="all, delete-orphan")


# ═══════════════════════════════════════════════════════════════════════════════
#  9. model_files
# ═══════════════════════════════════════════════════════════════════════════════

class ModelFile(Base):
    __tablename__ = "model_files"

    id          = Column(String(12), primary_key=True, default=_uuid)
    model_id    = Column(String(12), ForeignKey("trained_models.id", ondelete="CASCADE"), nullable=False)
    file_type   = Column(String(30), nullable=False)           # model | pipeline | scaler
    file_path   = Column(Text, nullable=False)
    file_size   = Column(Integer, default=0)
    created_at  = Column(DateTime, default=_utcnow)

    model = relationship("TrainedModel", back_populates="model_files")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. predictions
# ═══════════════════════════════════════════════════════════════════════════════

class Prediction(Base):
    __tablename__ = "predictions"

    id             = Column(String(12), primary_key=True, default=_uuid)
    experiment_id  = Column(String(12), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    model_id       = Column(String(12), ForeignKey("trained_models.id", ondelete="SET NULL"), nullable=True)
    input_data     = Column(JSON, nullable=False)
    prediction     = Column(JSON, nullable=False)              # single value or probabilities
    confidence     = Column(Float, nullable=True)
    created_at     = Column(DateTime, default=_utcnow)

    experiment = relationship("Experiment", back_populates="predictions")


# ═══════════════════════════════════════════════════════════════════════════════
# 11. evaluation_results
# ═══════════════════════════════════════════════════════════════════════════════

class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id             = Column(String(12), primary_key=True, default=_uuid)
    experiment_id  = Column(String(12), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    model_id       = Column(String(12), ForeignKey("trained_models.id", ondelete="SET NULL"), nullable=True)
    metric_name    = Column(String(50), nullable=False)
    metric_value   = Column(Float, nullable=False)
    split          = Column(String(20), default="test")        # train | test | cv
    created_at     = Column(DateTime, default=_utcnow)

    experiment = relationship("Experiment", back_populates="evaluations")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. comparison_runs
# ═══════════════════════════════════════════════════════════════════════════════

class ComparisonRun(Base):
    __tablename__ = "comparison_runs"

    id              = Column(String(12), primary_key=True, default=_uuid)
    dataset_id      = Column(String(12), nullable=False)
    task_type       = Column(String(30), nullable=False)
    target_column   = Column(String(255), nullable=False)
    algorithms      = Column(JSON, nullable=False)             # list[str]
    ranking_metric  = Column(String(50), nullable=False)       # metric used for ranking
    results         = Column(JSON, nullable=False)             # list of per-model dicts
    best_model_id   = Column(String(12), nullable=True)
    best_algorithm  = Column(String(50), default="")
    created_at      = Column(DateTime, default=_utcnow)


# ═══════════════════════════════════════════════════════════════════════════════
# 13. reports
# ═══════════════════════════════════════════════════════════════════════════════

class Report(Base):
    __tablename__ = "reports"

    id          = Column(String(12), primary_key=True, default=_uuid)
    dataset_id  = Column(String(12), nullable=True)
    experiment_id = Column(String(12), nullable=True)
    report_type = Column(String(30), nullable=False)           # validation | eda | experiment | full
    format      = Column(String(10), nullable=False)           # pdf | csv | json
    file_path   = Column(Text, nullable=False)
    file_size   = Column(Integer, default=0)
    created_at  = Column(DateTime, default=_utcnow)

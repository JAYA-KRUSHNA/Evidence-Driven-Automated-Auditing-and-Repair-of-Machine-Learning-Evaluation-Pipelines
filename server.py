"""
FastAPI Backend — Production-Ready REST API for the ML Experiment Auditor.

Production changes vs. initial version:
- GZipMiddleware for response compression
- Global exception handler (no raw stack traces to clients)
- DB sessions always closed in finally blocks (no connection leaks)
- Upload file size validation (max 200 MB)
- Connection pool_pre_ping=True + WAL mode for SQLite concurrency
- Input validation via Pydantic Field constraints
- /api/datasets/current endpoint to query loaded dataset state
- Proper try/except around all DB operations
- Thread-safe session handling (session per-request, never shared)
"""

import json
import logging
import sys
import threading
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

import pandas as pd
import numpy as np

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ARTIFACTS_DIR, get_software_versions, MAX_UPLOAD_SIZE_MB
from src.database.models import init_db, get_session
from src.database.repository import (
    DatasetRepository, ExperimentRepository,
    AuditRunRepository, FindingRepository,
    MetricRepository,
)
from src.data.dataset_catalog import DatasetCatalog
from src.data.dataset_loader import DatasetLoader, DatasetLoadError
from src.data.dataset_profiler import DatasetProfiler
from src.experiments.experiment_runner import ExperimentRunner, ExperimentConfig
from src.experiments.split_manager import SplitManager
from src.experiments.pipeline_builder import PipelineBuilder
from src.audit.audit_engine import AuditEngine
from src.repair.correction_engine import CorrectionEngine
from src.benchmark.benchmark_runner import BenchmarkRunner
from src.reports.report_builder import ReportBuilder

# ── Custom JSON encoder for numpy types ────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    """Converts numpy scalars/arrays to Python native types for JSON."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return round(float(obj), 6)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _safe_json(obj):
    """Round-trip through NumpyEncoder to get a pure-Python object."""
    return json.loads(json.dumps(obj, cls=NumpyEncoder, default=str))


# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Thread-safe session state ──────────────────────────────────────────────────
# Uses a per-process lock + dict so concurrent requests don't corrupt each other.
# In production with multiple workers, switch to Redis-backed sessions.

_state_lock = threading.Lock()
_session_data = {
    "current_df": None,
    "current_ds_name": "",
    "current_ds_id": "",
    "experiment_result": None,
    "experiment_config": None,
    "audit_result": None,
    "correction_result": None,
    "benchmark_result": None,
}


def _get_state(key):
    with _state_lock:
        return _session_data.get(key)


def _set_state(**kwargs):
    with _state_lock:
        _session_data.update(kwargs)


# ── App lifecycle ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Database initialized")
    yield
    logger.info("Server shutting down")


# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI ML Experiment Auditor",
    description="Evidence-Driven ML Pipeline Auditing Framework",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# GZip compress responses > 1 KB
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# Static frontend
STATIC_DIR = PROJECT_ROOT / "frontend"
STATIC_DIR.mkdir(exist_ok=True)

# ── Global exception handler ───────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}: {str(exc)[:200]}"},
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


# ── Pydantic Request Models ────────────────────────────────────────────────────

class ExperimentRequest(BaseModel):
    dataset_source: str = Field(..., min_length=1, max_length=50)
    dataset_source_id: str = Field(..., min_length=1, max_length=200)
    dataset_name: str = Field(..., min_length=1, max_length=255)
    target_column: str = Field(..., min_length=1, max_length=255)
    task_type: str = Field("classification", pattern="^(classification|regression)$")
    model_names: list[str] = Field(default=["random_forest"], min_length=1, max_length=6)
    test_size: float = Field(0.2, ge=0.05, le=0.5)
    cv_folds: int = Field(5, ge=2, le=10)
    random_seed: int = Field(42, ge=0, le=999999)
    experiment_name: str = Field("", max_length=255)

    @field_validator("model_names")
    @classmethod
    def validate_model_names(cls, v):
        valid = {
            "logistic_regression", "random_forest", "gradient_boosting",
            "decision_tree", "svm", "knn",
            "linear_regression", "random_forest_reg", "gradient_boosting_reg",
            "decision_tree_reg", "svr", "knn_reg",
        }
        for m in v:
            if m not in valid:
                raise ValueError(f"Unknown model '{m}'")
        return v


class BenchmarkRequest(BaseModel):
    dataset_source: str = Field(..., min_length=1, max_length=50)
    dataset_source_id: str = Field(..., min_length=1, max_length=200)
    dataset_name: str = Field(..., min_length=1, max_length=255)
    target_column: str = Field(..., min_length=1, max_length=255)
    task_type: str = Field("classification", pattern="^(classification|regression)$")
    model_name: str = Field("random_forest", max_length=50)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clean_metrics(metrics: dict) -> dict:
    """Make metrics JSON-serializable — drop confusion matrix, coerce numpy types."""
    clean = {}
    for k, v in metrics.items():
        if k == "confusion_matrix":
            continue
        try:
            clean[k] = round(float(v), 4)
        except (TypeError, ValueError):
            pass
    return clean


# ── Frontend ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>Frontend not found. Run from project root.</h1>", status_code=503)


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["System"])
async def health():
    return {"status": "ok", "versions": get_software_versions()}


# ── Dashboard ──────────────────────────────────────────────────────────────────

@app.get("/api/dashboard/stats", tags=["Dashboard"])
async def dashboard_stats():
    session = get_session()
    try:
        stats = {
            "datasets": DatasetRepository.count(session),
            "experiments": ExperimentRepository.count(session),
            "completed": ExperimentRepository.count(session, status="completed"),
            "audits": AuditRunRepository.count(session),
            "confirmed_findings": FindingRepository.count_confirmed(session),
            "suspected_findings": FindingRepository.count_suspected(session),
        }
        recent = ExperimentRepository.get_recent(session, limit=10)
        stats["recent_experiments"] = [
            {
                "id": e.id,
                "name": e.name,
                "status": e.status,
                "task_type": e.task_type,
                "target_column": e.target_column,
                "duration": e.duration_seconds,
                "created_at": str(e.created_at) if e.created_at else "",
            }
            for e in recent
        ]
        return stats
    except Exception as e:
        logger.error(f"dashboard_stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


# ── Datasets ───────────────────────────────────────────────────────────────────

@app.get("/api/datasets/catalog", tags=["Datasets"])
async def get_catalog(query: str = "", source: str = "", task_type: str = ""):
    try:
        catalog = DatasetCatalog()
        datasets = catalog.search(query=query, source=source, task_type=task_type)
        loader = DatasetLoader()
        return [
            {
                "name": d.name,
                "source": d.source,
                "source_id": d.source_id,
                "source_url": d.source_url,
                "task_type": d.task_type,
                "target_column": d.target_column,
                "row_count": d.row_count,
                "col_count": d.col_count,
                "description": d.description,
                "license_info": d.license_info,
                "missing_pct": d.missing_pct,
                "is_cached": loader.is_cached(d.source, d.source_id),
            }
            for d in datasets
        ]
    except Exception as e:
        logger.error(f"get_catalog error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/datasets/load", tags=["Datasets"])
async def load_dataset(source: str, source_id: str, name: str = ""):
    loader = DatasetLoader()
    session = get_session()
    try:
        df, metadata = loader.load(source, source_id, name)

        _set_state(
            current_df=df,
            current_ds_name=name or metadata.get("name", ""),
            current_ds_id=f"{source}_{source_id}",
        )

        # Register in DB
        existing = DatasetRepository.get_by_source_id(session, source, source_id)
        if not existing:
            db_ds = DatasetRepository.create(
                session, name=name or metadata.get("name", source_id),
                source=source, source_id=source_id,
                source_url=metadata.get("source_url", ""),
                task_type=metadata.get("task_type", ""),
                target_column=metadata.get("target_column", ""),
                row_count=len(df), col_count=len(df.columns),
                missing_pct=round(float(df.isna().mean().mean() * 100), 2),
                description=metadata.get("description", ""),
            )
            _set_state(current_ds_id=db_ds.id)
        else:
            _set_state(current_ds_id=existing.id)

        profiler = DatasetProfiler()
        target_col = metadata.get("target_column", "")
        profile = profiler.profile(df, target_col)

        return _safe_json({
            "status": "ok",
            "rows": len(df),
            "cols": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "preview": df.head(30).to_dict(orient="records"),
            "profile": profile.to_dict(),
            "target_candidates": profile.target_candidates,
        })
    except DatasetLoadError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"load_dataset error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@app.post("/api/datasets/upload", tags=["Datasets"])
async def upload_dataset(request: Request, file: UploadFile = File(...)):
    # Validate file size before reading
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_SIZE_MB} MB."
        )

    # Validate extension
    fname = file.filename or ""
    suffix = Path(fname).suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'. Use .csv or .parquet.")

    loader = DatasetLoader()
    try:
        df, metadata = loader.load_from_upload(file.file)

        # Sanitize column names
        df.columns = [
            str(c).strip().replace(" ", "_").replace("-", "_")
            for c in df.columns
        ]

        _set_state(
            current_df=df,
            current_ds_name=fname,
            current_ds_id=metadata.get("source_id", "uploaded"),
        )

        profiler = DatasetProfiler()
        profile = profiler.profile(df)

        return _safe_json({
            "status": "ok",
            "rows": len(df),
            "cols": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "preview": df.head(30).to_dict(orient="records"),
            "profile": profile.to_dict(),
            "target_candidates": profile.target_candidates,
        })
    except Exception as e:
        logger.error(f"upload_dataset error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/datasets/current", tags=["Datasets"])
async def get_current_dataset():
    """Return info about the currently loaded dataset."""
    df = _get_state("current_df")
    if df is None:
        return {"loaded": False}
    return _safe_json({
        "loaded": True,
        "name": _get_state("current_ds_name"),
        "id": _get_state("current_ds_id"),
        "rows": len(df),
        "cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_pct": round(float(df.isna().mean().mean() * 100), 2),
        "numeric_cols": int(df.select_dtypes(include='number').shape[1]),
        "categorical_cols": int(df.select_dtypes(exclude='number').shape[1]),
    })


@app.get("/api/datasets/current/preview", tags=["Datasets"])
async def get_current_preview(rows: int = 50):
    """Return a preview of the loaded dataset with column stats."""
    df = _get_state("current_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset loaded.")
    rows = max(5, min(rows, 200))
    preview = df.head(rows).to_dict(orient="records")
    col_stats = {}
    for col in df.columns:
        s = df[col]
        stat = {
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 1),
            "unique": int(s.nunique()),
        }
        if pd.api.types.is_numeric_dtype(s):
            stat["min"] = round(float(s.min()), 4) if not s.isna().all() else None
            stat["max"] = round(float(s.max()), 4) if not s.isna().all() else None
            stat["mean"] = round(float(s.mean()), 4) if not s.isna().all() else None
            stat["std"] = round(float(s.std()), 4) if not s.isna().all() else None
        col_stats[col] = stat
    return _safe_json({
        "rows": len(df),
        "cols": len(df.columns),
        "preview": preview,
        "col_stats": col_stats,
    })


@app.get("/api/datasets/current/profile", tags=["Datasets"])
async def get_current_profile(target_column: str = ""):
    df = _get_state("current_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset loaded. Load a dataset first.")

    if target_column and target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{target_column}' not found in dataset.")

    try:
        profiler = DatasetProfiler()
        profile = profiler.profile(df, target_column)
        target_analysis = profiler.get_target_analysis(df, target_column) if target_column else {}

        return _safe_json({
            "profile": profile.to_dict(),
            "target_analysis": target_analysis,
            "target_candidates": profile.target_candidates,
        })
    except Exception as e:
        logger.error(f"get_current_profile error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Experiments ────────────────────────────────────────────────────────────────

@app.get("/api/experiments/models", tags=["Experiments"])
async def get_available_models(task_type: str = "classification"):
    if task_type not in ("classification", "regression"):
        raise HTTPException(status_code=400, detail="task_type must be 'classification' or 'regression'")
    models = PipelineBuilder.get_available_models(task_type)
    return {"models": models}


@app.get("/api/experiments/history", tags=["Experiments"])
async def get_experiment_history(limit: int = 20):
    """Return all past experiments from DB with their key metrics."""
    session = get_session()
    try:
        experiments = ExperimentRepository.get_recent(session, limit=max(1, min(limit, 100)))
        history = []
        for exp in experiments:
            metrics = MetricRepository.get_as_dict(session, exp.id, split_type="test")
            history.append({
                "id": exp.id,
                "name": exp.name,
                "status": exp.status,
                "task_type": exp.task_type,
                "target_column": exp.target_column,
                "duration": exp.duration_seconds,
                "created_at": str(exp.created_at) if exp.created_at else "",
                "is_corrected": exp.is_corrected,
                "test_metrics": {k: round(v, 4) for k, v in metrics.items()},
                "model_config": exp.get_model_config(),
                "split_config": exp.get_split_config(),
            })
        return history
    except Exception as e:
        logger.error(f"get_experiment_history error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@app.post("/api/experiments/run", tags=["Experiments"])
async def run_experiment(req: ExperimentRequest):
    try:
        # Load dataset if not already in memory
        df = _get_state("current_df")
        if df is None:
            loader = DatasetLoader()
            try:
                df, _ = loader.load(req.dataset_source, req.dataset_source_id, req.dataset_name)
                _set_state(current_df=df, current_ds_name=req.dataset_name)
            except DatasetLoadError as e:
                raise HTTPException(status_code=400, detail=f"Cannot load dataset: {e}")

        # Validate target column
        if req.target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{req.target_column}' not found. Available columns: {list(df.columns[:10])}"
            )

        config = ExperimentConfig(
            dataset_id=_get_state("current_ds_id") or "",
            target_column=req.target_column,
            task_type=req.task_type,
            model_names=req.model_names,
            test_size=req.test_size,
            cv_folds=req.cv_folds,
            random_seed=req.random_seed,
            experiment_name=req.experiment_name or f"{req.dataset_name}_exp",
        )

        runner = ExperimentRunner()
        result = runner.run(df, config)

        _set_state(experiment_result=result, experiment_config=config)

        if result.status == "failed":
            return {"status": "failed", "error": result.error}

        # Format training results — expose ALL computed data
        training_results = {}
        for model_name, tr in result.training_results.items():
            fi = {}
            if tr.feature_importances:
                for feat, imp in tr.feature_importances.items():
                    try:
                        fi[str(feat).replace('num__', '').replace('cat__', '')] = round(float(imp), 6)
                    except (TypeError, ValueError):
                        pass

            # Extract confusion matrix separately (list of lists, safe to pass)
            confusion_matrix = tr.test_metrics.get("confusion_matrix", None)

            # Build per-fold CV table grouped by fold number
            fold_table = {}
            for entry in (tr.cv_fold_scores or []):
                fold = entry["fold"]
                if fold not in fold_table:
                    fold_table[fold] = {}
                fold_table[fold][entry["metric"]] = round(float(entry["value"]), 4)
            cv_folds_list = [
                {"fold": f, **metrics}
                for f, metrics in sorted(fold_table.items())
            ]

            training_results[model_name] = _safe_json({
                "model_name": tr.model_name,
                "train_metrics": _clean_metrics(tr.train_metrics),
                "test_metrics": _clean_metrics(tr.test_metrics),
                "cv_metrics": {k: round(float(v), 4) for k, v in tr.cv_metrics.items()},
                "cv_folds": cv_folds_list,
                "baseline_metrics": _clean_metrics(tr.baseline_metrics),
                "training_time": round(float(tr.training_time), 3),
                "feature_importances": fi,
                "confusion_matrix": confusion_matrix,
            })

        # Prepare data for audit
        df_clean = df.dropna(subset=[req.target_column])
        X = df_clean.drop(columns=[req.target_column])
        y = df_clean[req.target_column]

        if req.task_type == "classification":
            y, _ = PipelineBuilder.encode_target(y, "classification")

        X_train, X_test, y_train, y_test = SplitManager.train_test_split(
            X, y, test_size=req.test_size, random_seed=req.random_seed, task_type=req.task_type,
        )

        first_model = list(result.training_results.values())[0]

        # Run audit — gracefully degrade if it fails
        try:
            audit_engine = AuditEngine()
            audit_result = audit_engine.run_full_audit(
                df=df,
                target_column=req.target_column,
                task_type=req.task_type,
                experiment_id=result.experiment_id,
                X_train=X_train, X_test=X_test,
                y_train=y_train, y_test=y_test,
                train_metrics=first_model.train_metrics,
                test_metrics=first_model.test_metrics,
                cv_metrics=first_model.cv_metrics,
                baseline_metrics=first_model.baseline_metrics,
                feature_importances=first_model.feature_importances,
                target_analysis=result.target_analysis,
                pipeline=first_model.pipeline,
            )
            _set_state(audit_result=audit_result)

            findings = []
            for f in audit_result.findings:
                try:
                    findings.append(_safe_json(f.to_dict()))
                except Exception:
                    pass

            audit_payload = {
                "summary": audit_result.summary(),
                "findings": findings,
                "by_category": {k: len(v) for k, v in audit_result.findings_by_category().items()},
                "by_severity": {k: len(v) for k, v in audit_result.findings_by_severity().items()},
            }
        except Exception as audit_err:
            logger.error(f"Audit phase failed: {audit_err}", exc_info=True)
            audit_payload = {
                "summary": {
                    "total_findings": 0, "confirmed": 0, "suspected": 0,
                    "auto_correctable": 0, "critical": 0, "high": 0,
                    "medium": 0, "low": 0, "duration_seconds": 0,
                    "error": str(audit_err),
                },
                "findings": [], "by_category": {}, "by_severity": {},
            }

        return _safe_json({
            "status": "completed",
            "experiment_id": result.experiment_id,
            "total_time": result.total_time,
            "split_info": result.split_info,
            "training_results": training_results,
            "audit": audit_payload,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"run_experiment unhandled: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")



@app.get("/api/findings/export", tags=["Findings"])
async def export_findings_csv():
    """Export all current audit findings as a CSV file."""
    import io, csv
    audit_result = _get_state("audit_result")
    if not audit_result or not audit_result.findings:
        raise HTTPException(status_code=404, detail="No findings to export. Run an experiment first.")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "finding_id", "category", "severity", "status", "confidence",
        "auto_correctable", "description", "affected_columns",
        "recommendation", "resolution_status"
    ])
    writer.writeheader()
    for f in audit_result.findings:
        d = f.to_dict()
        writer.writerow({
            "finding_id": d.get("finding_id", ""),
            "category": d.get("category", ""),
            "severity": d.get("severity", ""),
            "status": d.get("status", ""),
            "confidence": round(float(d.get("confidence") or 0), 4),
            "auto_correctable": d.get("auto_correctable", False),
            "description": d.get("description", ""),
            "affected_columns": ", ".join(d.get("affected_columns") or []),
            "recommendation": d.get("recommendation", ""),
            "resolution_status": d.get("resolution_status", "open"),
        })
    csv_bytes = output.getvalue().encode()
    from fastapi.responses import Response
    ds_name = (_get_state("current_ds_name") or "audit").replace(" ", "_").lower()
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={ds_name}_findings.csv"}
    )


# ── Corrections ────────────────────────────────────────────────────────────────

@app.post("/api/corrections/apply", tags=["Corrections"])
async def apply_corrections():
    audit_result = _get_state("audit_result")
    experiment_result = _get_state("experiment_result")
    experiment_config = _get_state("experiment_config")
    df = _get_state("current_df")

    if not audit_result:
        raise HTTPException(status_code=400, detail="No audit result found. Run an experiment first.")
    if df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")
    if not experiment_config:
        raise HTTPException(status_code=400, detail="No experiment config found.")

    try:
        engine = CorrectionEngine()
        split_strategy = getattr(experiment_config, "split_strategy", "auto")
        plan = engine.propose_corrections(
            audit_result, df,
            experiment_config.target_column,
            split_config={"strategy": split_strategy},
        )

        if not plan.has_corrections:
            return {"status": "no_corrections", "message": "No auto-correctable issues found in this pipeline."}

        correction_result = engine.apply_corrections(
            plan=plan, df=df, config=experiment_config,
            original_result=experiment_result,
            original_audit=audit_result,
        )

        _set_state(correction_result=correction_result)

        return _safe_json({
            "status": correction_result.status,
            "error": correction_result.error,
            "actions": [a.to_dict() for a in plan.actions],
            "before_metrics": _clean_metrics(correction_result.before_metrics),
            "after_metrics": _clean_metrics(correction_result.after_metrics),
            "resolved": len(correction_result.resolved_findings),
            "unresolved": len(correction_result.unresolved_findings),
        })
    except Exception as e:
        logger.error(f"apply_corrections error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


# ── Benchmark ──────────────────────────────────────────────────────────────────

@app.post("/api/benchmark/run", tags=["Benchmark"])
async def run_benchmark(req: BenchmarkRequest):
    try:
        loader = DatasetLoader()
        df, _ = loader.load(req.dataset_source, req.dataset_source_id, req.dataset_name)

        if req.target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{req.target_column}' not found.")

        runner = BenchmarkRunner()
        result = runner.run_benchmark(df, req.target_column, req.task_type, req.model_name)
        _set_state(benchmark_result=result)

        scenarios = []
        for r in result.scenario_results:
            scenarios.append(_safe_json({
                "name": r.scenario_name,
                "type": r.corruption_type,
                "ground_truth": r.ground_truth,
                "detected": r.detected_categories,
                "true_positives": r.true_positives,
                "false_positives": r.false_positives,
                "false_negatives": r.false_negatives,
                "precision": round(r.precision, 4),
                "recall": round(r.recall, 4),
                "f1": round(r.f1, 4),
                "is_holdout": r.is_holdout,
                "error": r.error,
            }))

        return _safe_json({
            "status": "completed",
            "overall": result.to_dict(),
            "scenarios": scenarios,
        })
    except HTTPException:
        raise
    except DatasetLoadError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"run_benchmark error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


# ── Reports ────────────────────────────────────────────────────────────────────

@app.post("/api/reports/generate", tags=["Reports"])
async def generate_report(format: str = "json"):
    if format not in ("json", "html"):
        raise HTTPException(status_code=400, detail="format must be 'json' or 'html'")

    experiment_result = _get_state("experiment_result")
    audit_result = _get_state("audit_result")
    experiment_config = _get_state("experiment_config")

    if not experiment_result or not audit_result:
        raise HTTPException(status_code=400, detail="Run an experiment first before generating a report.")

    try:
        builder = ReportBuilder()

        profile = getattr(experiment_result, "profile", None) or {}
        dataset_info = {
            "name": _get_state("current_ds_name") or "",
            "row_count": profile.get("row_count", 0),
            "col_count": profile.get("col_count", 0),
        }

        exp_results = {}
        for model_name, tr in experiment_result.training_results.items():
            exp_results[model_name] = {
                "test_metrics": _clean_metrics(tr.test_metrics),
                "train_metrics": _clean_metrics(tr.train_metrics),
                "cv_metrics": {k: round(float(v), 4) for k, v in tr.cv_metrics.items()},
            }

        audit_summary = audit_result.summary()
        audit_summary["findings"] = [_safe_json(f.to_dict()) for f in audit_result.findings]

        cr = _get_state("correction_result")
        correction_summary = cr.comparison_summary() if cr and cr.status == "completed" else None

        br = _get_state("benchmark_result")
        benchmark_summary = br.to_dict() if br else None

        report = builder.build_report(
            dataset_info=dataset_info,
            experiment_config=experiment_config.to_dict() if experiment_config else {},
            experiment_result=exp_results,
            audit_result=audit_summary,
            correction_result=correction_summary,
            benchmark_result=benchmark_summary,
        )

        if format == "html":
            filepath = builder.export_html(report)
            return FileResponse(
                filepath, media_type="text/html",
                filename=filepath.name,
                headers={"Content-Disposition": f"attachment; filename={filepath.name}"},
            )
        else:
            filepath = builder.export_json(report)
            return FileResponse(
                filepath, media_type="application/json",
                filename=filepath.name,
                headers={"Content-Disposition": f"attachment; filename={filepath.name}"},
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"generate_report error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


# ── Static files (must be last) ────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

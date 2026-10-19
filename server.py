"""
FastAPI Backend — REST API for the ML Experiment Auditor.

Serves both the API endpoints and the static frontend files.
All ML operations happen server-side; the frontend communicates via JSON.
"""

import json
import logging
import sys
import asyncio
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ARTIFACTS_DIR, get_software_versions


# ── Custom JSON encoder for numpy types ──────────────────────────────────────

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
# In-memory store for current session data
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Database initialized")
    yield


app = FastAPI(
    title="AI ML Experiment Auditor",
    description="Evidence-Driven ML Pipeline Auditing Framework",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
STATIC_DIR = PROJECT_ROOT / "frontend"
STATIC_DIR.mkdir(exist_ok=True)

# ── Pydantic Models ───────────────────────────────────────────────────────────

class ExperimentRequest(BaseModel):
    dataset_source: str
    dataset_source_id: str
    dataset_name: str
    target_column: str
    task_type: str = "classification"
    model_names: list[str] = ["random_forest"]
    test_size: float = 0.2
    cv_folds: int = 5
    random_seed: int = 42
    experiment_name: str = ""


class BenchmarkRequest(BaseModel):
    dataset_source: str
    dataset_source_id: str
    dataset_name: str
    target_column: str
    task_type: str = "classification"
    model_name: str = "random_forest"


# ── Frontend Route ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>Frontend not built yet</h1>")


# ── API Routes ────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "versions": get_software_versions()}


# ── Dashboard Stats ───────────────────────────────────────────────────────────

@app.get("/api/dashboard/stats")
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
    finally:
        session.close()


# ── Dataset Routes ────────────────────────────────────────────────────────────

@app.get("/api/datasets/catalog")
async def get_catalog(query: str = "", source: str = "", task_type: str = ""):
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


@app.post("/api/datasets/load")
async def load_dataset(source: str, source_id: str, name: str = ""):
    loader = DatasetLoader()
    try:
        df, metadata = loader.load(source, source_id, name)

        # Store in memory
        _session_data["current_df"] = df
        _session_data["current_ds_name"] = name or metadata.get("name", "")
        _session_data["current_ds_id"] = f"{source}_{source_id}"

        # Register in DB
        session = get_session()
        existing = DatasetRepository.get_by_source_id(session, source, source_id)
        if not existing:
            db_ds = DatasetRepository.create(
                session, name=name, source=source, source_id=source_id,
                source_url=metadata.get("source_url", ""),
                task_type=metadata.get("task_type", ""),
                target_column=metadata.get("target_column", ""),
                row_count=len(df), col_count=len(df.columns),
                missing_pct=round(df.isna().mean().mean() * 100, 2),
                description=metadata.get("description", ""),
            )
            _session_data["current_ds_id"] = db_ds.id
        else:
            _session_data["current_ds_id"] = existing.id
        session.close()

        # Profile
        profiler = DatasetProfiler()
        target_col = metadata.get("target_column", "")
        profile = profiler.profile(df, target_col)

        return {
            "status": "ok",
            "rows": len(df),
            "cols": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "preview": json.loads(df.head(30).to_json(orient="records", default_handler=str)),
            "profile": profile.to_dict(),
            "target_candidates": profile.target_candidates,
        }
    except DatasetLoadError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    loader = DatasetLoader()
    try:
        df, metadata = loader.load_from_upload(file.file)

        _session_data["current_df"] = df
        _session_data["current_ds_name"] = file.filename or "uploaded"
        _session_data["current_ds_id"] = metadata.get("source_id", "uploaded")

        profiler = DatasetProfiler()
        profile = profiler.profile(df)

        return {
            "status": "ok",
            "rows": len(df),
            "cols": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "preview": json.loads(df.head(30).to_json(orient="records", default_handler=str)),
            "profile": profile.to_dict(),
            "target_candidates": profile.target_candidates,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/datasets/current/profile")
async def get_current_profile(target_column: str = ""):
    df = _session_data.get("current_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset loaded")

    profiler = DatasetProfiler()
    profile = profiler.profile(df, target_column)
    target_analysis = profiler.get_target_analysis(df, target_column) if target_column else {}

    return {
        "profile": profile.to_dict(),
        "target_analysis": target_analysis,
        "target_candidates": profile.target_candidates,
    }


# ── Experiment Routes ─────────────────────────────────────────────────────────

@app.get("/api/experiments/models")
async def get_available_models(task_type: str = "classification"):
    models = PipelineBuilder.get_available_models(task_type)
    return {"models": models}


@app.post("/api/experiments/run")
async def run_experiment(req: ExperimentRequest):
    # Load dataset if not already loaded
    df = _session_data.get("current_df")
    if df is None:
        loader = DatasetLoader()
        try:
            df, _ = loader.load(req.dataset_source, req.dataset_source_id, req.dataset_name)
            _session_data["current_df"] = df
            _session_data["current_ds_name"] = req.dataset_name
        except DatasetLoadError as e:
            raise HTTPException(status_code=400, detail=f"Cannot load dataset: {e}")

    config = ExperimentConfig(
        dataset_id=_session_data.get("current_ds_id", ""),
        target_column=req.target_column,
        task_type=req.task_type,
        model_names=req.model_names,
        test_size=req.test_size,
        cv_folds=req.cv_folds,
        random_seed=req.random_seed,
        experiment_name=req.experiment_name or f"{req.dataset_name}_experiment",
    )

    runner = ExperimentRunner()
    result = runner.run(df, config)

    _session_data["experiment_result"] = result
    _session_data["experiment_config"] = config

    if result.status == "failed":
        return {"status": "failed", "error": result.error}

    # Format training results
    training_results = {}
    for model_name, tr in result.training_results.items():
        # Clean feature importances (may contain numpy float32)
        fi = {}
        if tr.feature_importances:
            for feat, imp in tr.feature_importances.items():
                try:
                    fi[str(feat)] = round(float(imp), 6)
                except (TypeError, ValueError):
                    pass

        training_results[model_name] = _safe_json({
            "model_name": tr.model_name,
            "train_metrics": _clean_metrics(tr.train_metrics),
            "test_metrics": _clean_metrics(tr.test_metrics),
            "cv_metrics": {k: round(float(v), 4) for k, v in tr.cv_metrics.items()},
            "baseline_metrics": _clean_metrics(tr.baseline_metrics),
            "training_time": round(float(tr.training_time), 3),
            "feature_importances": fi,
        })

    # Run audit
    audit_engine = AuditEngine()

    df_clean = df.dropna(subset=[req.target_column])
    X = df_clean.drop(columns=[req.target_column])
    y = df_clean[req.target_column]

    if req.task_type == "classification":
        y, _ = PipelineBuilder.encode_target(y, "classification")

    X_train, X_test, y_train, y_test = SplitManager.train_test_split(
        X, y, test_size=req.test_size, random_seed=req.random_seed, task_type=req.task_type,
    )

    first_model = list(result.training_results.values())[0]

    try:
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
    except Exception as audit_err:
        logger.error(f"Audit failed: {audit_err}", exc_info=True)
        # Return training results even if audit fails
        return _safe_json({
            "status": "completed_no_audit",
            "experiment_id": result.experiment_id,
            "total_time": result.total_time,
            "split_info": result.split_info,
            "training_results": training_results,
            "audit": {"summary": {"total_findings": 0, "confirmed": 0, "suspected": 0,
                                   "auto_correctable": 0, "critical": 0, "high": 0,
                                   "medium": 0, "low": 0, "duration_seconds": 0},
                      "findings": [], "by_category": {}, "by_severity": {},
                      "error": str(audit_err)},
        })

    _session_data["audit_result"] = audit_result

    # Format findings — safe-convert each one
    findings = []
    for f in audit_result.findings:
        try:
            findings.append(_safe_json(f.to_dict()))
        except Exception:
            pass

    return _safe_json({
        "status": "completed",
        "experiment_id": result.experiment_id,
        "total_time": result.total_time,
        "split_info": result.split_info,
        "training_results": training_results,
        "audit": {
            "summary": audit_result.summary(),
            "findings": findings,
            "by_category": {
                k: len(v) for k, v in audit_result.findings_by_category().items()
            },
            "by_severity": {
                k: len(v) for k, v in audit_result.findings_by_severity().items()
            },
        },
    })


# ── Correction Routes ─────────────────────────────────────────────────────────

@app.post("/api/corrections/apply")
async def apply_corrections():
    audit_result = _session_data.get("audit_result")
    experiment_result = _session_data.get("experiment_result")
    experiment_config = _session_data.get("experiment_config")
    df = _session_data.get("current_df")

    if not all([audit_result, experiment_result, experiment_config, df is not None]):
        raise HTTPException(status_code=400, detail="Run an experiment and audit first")

    engine = CorrectionEngine()

    plan = engine.propose_corrections(
        audit_result, df,
        experiment_config.target_column,
        split_config={"strategy": experiment_config.split_strategy},
    )

    if not plan.has_corrections:
        return {"status": "no_corrections", "message": "No auto-correctable issues found"}

    correction_result = engine.apply_corrections(
        plan=plan, df=df, config=experiment_config,
        original_result=experiment_result,
        original_audit=audit_result,
    )

    _session_data["correction_result"] = correction_result

    return {
        "status": correction_result.status,
        "error": correction_result.error,
        "actions": [a.to_dict() for a in plan.actions],
        "before_metrics": _clean_metrics(correction_result.before_metrics),
        "after_metrics": _clean_metrics(correction_result.after_metrics),
        "resolved": len(correction_result.resolved_findings),
        "unresolved": len(correction_result.unresolved_findings),
    }


# ── Benchmark Routes ──────────────────────────────────────────────────────────

@app.post("/api/benchmark/run")
async def run_benchmark(req: BenchmarkRequest):
    loader = DatasetLoader()
    try:
        df, _ = loader.load(req.dataset_source, req.dataset_source_id, req.dataset_name)
    except DatasetLoadError as e:
        raise HTTPException(status_code=400, detail=f"Cannot load dataset: {e}")

    if req.target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.target_column}' not found")

    runner = BenchmarkRunner()
    result = runner.run_benchmark(
        df, req.target_column, req.task_type, req.model_name,
    )

    _session_data["benchmark_result"] = result

    scenarios = []
    for r in result.scenario_results:
        scenarios.append({
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
        })

    return {
        "status": "completed",
        "overall": result.to_dict(),
        "scenarios": scenarios,
    }


# ── Report Routes ─────────────────────────────────────────────────────────────

@app.post("/api/reports/generate")
async def generate_report(format: str = "json"):
    experiment_result = _session_data.get("experiment_result")
    audit_result = _session_data.get("audit_result")
    experiment_config = _session_data.get("experiment_config")

    if not experiment_result or not audit_result:
        raise HTTPException(status_code=400, detail="Run an experiment first")

    builder = ReportBuilder()

    dataset_info = {
        "name": _session_data.get("current_ds_name", ""),
        "row_count": experiment_result.profile.get("row_count", 0) if experiment_result.profile else 0,
        "col_count": experiment_result.profile.get("col_count", 0) if experiment_result.profile else 0,
    }

    exp_results = {}
    for model_name, tr in experiment_result.training_results.items():
        exp_results[model_name] = {
            "test_metrics": _clean_metrics(tr.test_metrics),
            "train_metrics": _clean_metrics(tr.train_metrics),
            "cv_metrics": {k: round(v, 4) for k, v in tr.cv_metrics.items()},
        }

    audit_summary = audit_result.summary()
    audit_summary["findings"] = [f.to_dict() for f in audit_result.findings]

    correction_summary = None
    cr = _session_data.get("correction_result")
    if cr and cr.status == "completed":
        correction_summary = cr.comparison_summary()

    benchmark_summary = None
    br = _session_data.get("benchmark_result")
    if br:
        benchmark_summary = br.to_dict()

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
        return FileResponse(filepath, media_type="text/html", filename=filepath.name)
    else:
        filepath = builder.export_json(report)
        return FileResponse(filepath, media_type="application/json", filename=filepath.name)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_metrics(metrics: dict) -> dict:
    """Make metrics JSON-serializable."""
    clean = {}
    for k, v in metrics.items():
        if k == "confusion_matrix":
            continue
        if isinstance(v, (int, float)):
            clean[k] = round(float(v), 4)
        elif isinstance(v, np.floating):
            clean[k] = round(float(v), 4)
        elif isinstance(v, np.integer):
            clean[k] = int(v)
    return clean


# ── Mount static files (must be last) ────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

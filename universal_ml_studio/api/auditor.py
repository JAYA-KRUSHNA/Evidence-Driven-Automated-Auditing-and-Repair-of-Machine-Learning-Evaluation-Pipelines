"""
Universal AI/ML Studio — Pipeline Auditor, Leakage Detection, and Repair Router.

Integrates the Evidence-Driven Automated Pipeline Auditor, Leakage Detection,
Correction/Repair Engine, and Benchmark Lab into Universal AI/ML Studio.
"""

import csv
import io
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel
import pandas as pd

from ..dataset.registry import DatasetRegistry, DatasetNotFoundError
from ..eda.target import suggest_target_columns

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auditor", tags=["Pipeline Auditor"])

# In-memory auditor state cache
_auditor_state: Dict[str, Any] = {
    "current_dataset_id": None,
    "current_df": None,
    "experiment_config": None,
    "experiment_result": None,
    "audit_result": None,
    "correction_result": None,
    "benchmark_result": None,
}


def _clean_val(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and (pd.isna(v) or str(v).lower() in ("nan", "inf", "-inf")):
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


def _clean_dict(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _clean_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_clean_dict(v) for v in d]
    return _clean_val(d)


class BenchmarkRequest(BaseModel):
    dataset_id: Optional[str] = None
    target_column: Optional[str] = None
    task_type: Optional[str] = "classification"
    model_name: Optional[str] = "random_forest"


@router.post("/audit/{dataset_id}")
async def run_pipeline_audit(
    dataset_id: str,
    target_column: Optional[str] = Query(None, description="Target column name"),
    task_type: Optional[str] = Query(None, description="classification or regression"),
    model_name: Optional[str] = Query("random_forest", description="Base model name"),
):
    """Run full evidence-driven audit on dataset to detect leakage and evaluation flaws."""
    try:
        df = DatasetRegistry.load_dataframe(dataset_id)
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty.")

    # Auto-detect target column if not provided
    if not target_column:
        suggested = suggest_target_columns(df)
        target_column = suggested[0] if suggested else df.columns[-1]

    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found.")

    # Auto-detect task type
    if not task_type:
        y = df[target_column].dropna()
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15:
            task_type = "regression"
        else:
            task_type = "classification"

    from src.experiments.experiment_runner import ExperimentRunner, ExperimentConfig
    from src.experiments.split_manager import SplitManager
    from src.experiments.pipeline_builder import PipelineBuilder
    from src.audit.audit_engine import AuditEngine

    try:
        config = ExperimentConfig(
            dataset_id=dataset_id,
            target_column=target_column,
            task_type=task_type,
            model_names=[model_name],
            test_size=0.2,
            random_seed=42,
            experiment_name=f"audit_{dataset_id}",
        )

        runner = ExperimentRunner()
        result = runner.run(df, config)

        if result.status == "failed":
            raise HTTPException(status_code=500, detail=f"Experiment runner failed: {result.error}")

        # Prepare splits for deep audit
        df_clean = df.dropna(subset=[target_column])
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]

        if task_type == "classification":
            y, _ = PipelineBuilder.encode_target(y, "classification")

        X_train, X_test, y_train, y_test = SplitManager.train_test_split(
            X, y, test_size=0.2, random_seed=42, task_type=task_type,
        )

        first_model = list(result.training_results.values())[0]

        audit_engine = AuditEngine()
        audit_result = audit_engine.run_full_audit(
            df=df,
            target_column=target_column,
            task_type=task_type,
            experiment_id=result.experiment_id,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            train_metrics=first_model.train_metrics,
            test_metrics=first_model.test_metrics,
            cv_metrics=first_model.cv_metrics,
            baseline_metrics=first_model.baseline_metrics,
            feature_importances=first_model.feature_importances,
            target_analysis=result.target_analysis,
            pipeline=first_model.pipeline,
        )

        # Update cache
        _auditor_state.update({
            "current_dataset_id": dataset_id,
            "current_df": df,
            "experiment_config": config,
            "experiment_result": result,
            "audit_result": audit_result,
        })

        findings = []
        for f in audit_result.findings:
            findings.append(_clean_dict(f.to_dict()))

        # Calculate Pipeline Integrity Score (0-100)
        penalty = (
            audit_result.critical_count * 25
            + audit_result.high_count * 15
            + audit_result.suspected_count * 5
        )
        integrity_score = max(0, 100 - penalty)

        return _clean_dict({
            "status": "completed",
            "dataset_id": dataset_id,
            "target_column": target_column,
            "task_type": task_type,
            "model_name": model_name,
            "integrity_score": integrity_score,
            "summary": audit_result.summary(),
            "findings": findings,
            "by_category": {k: len(v) for k, v in audit_result.findings_by_category().items()},
            "by_severity": {k: len(v) for k, v in audit_result.findings_by_severity().items()},
            "baseline_metrics": first_model.test_metrics,
            "train_metrics": first_model.train_metrics,
        })
    except Exception as e:
        logger.error(f"Audit execution error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audit failed: {type(e).__name__}: {str(e)}")


@router.post("/repair/{dataset_id}")
async def repair_pipeline(dataset_id: str):
    """Propose and apply evidence-driven corrections to repair evaluation pipeline."""
    audit_result = _auditor_state.get("audit_result")
    experiment_result = _auditor_state.get("experiment_result")
    experiment_config = _auditor_state.get("experiment_config")
    df = _auditor_state.get("current_df")

    # If audit has not been run on this dataset yet, automatically run it first
    if audit_result is None or _auditor_state.get("current_dataset_id") != dataset_id:
        await run_pipeline_audit(dataset_id)
        audit_result = _auditor_state.get("audit_result")
        experiment_result = _auditor_state.get("experiment_result")
        experiment_config = _auditor_state.get("experiment_config")
        df = _auditor_state.get("current_df")

    if not audit_result or df is None or not experiment_config:
        raise HTTPException(status_code=400, detail="Cannot repair: no active audit state found.")

    from src.repair.correction_engine import CorrectionEngine

    try:
        engine = CorrectionEngine()
        split_strategy = getattr(experiment_config, "split_strategy", "auto")
        plan = engine.propose_corrections(
            audit_result,
            df,
            experiment_config.target_column,
            split_config={"strategy": split_strategy},
        )

        if not plan.has_corrections:
            return {
                "status": "no_corrections",
                "message": "No auto-correctable pipeline defects detected. Pipeline evaluation is sound.",
                "actions": [],
                "before_metrics": {},
                "after_metrics": {},
                "resolved": 0,
                "unresolved": 0,
            }

        correction_result = engine.apply_corrections(
            plan=plan,
            df=df,
            config=experiment_config,
            original_result=experiment_result,
            original_audit=audit_result,
        )

        _auditor_state["correction_result"] = correction_result

        return _clean_dict({
            "status": correction_result.status,
            "message": "Pipeline successfully repaired with evidence-driven corrections.",
            "actions": [a.to_dict() for a in plan.actions],
            "before_metrics": correction_result.before_metrics,
            "after_metrics": correction_result.after_metrics,
            "resolved": len(correction_result.resolved_findings),
            "unresolved": len(correction_result.unresolved_findings),
        })
    except Exception as e:
        logger.error(f"Repair execution error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline repair failed: {type(e).__name__}: {str(e)}")


@router.post("/benchmark")
async def run_benchmark(req: BenchmarkRequest):
    """Run synthetic leakage benchmark testing auditor against ground truth corruptions."""
    from src.benchmark.benchmark_runner import BenchmarkRunner

    try:
        if req.dataset_id:
            df = DatasetRegistry.load_dataframe(req.dataset_id)
        else:
            # Fallback to current or first dataset
            if _auditor_state.get("current_df") is not None:
                df = _auditor_state.get("current_df")
            else:
                datasets = DatasetRegistry.list_all()
                if not datasets:
                    raise HTTPException(status_code=400, detail="No datasets available for benchmark.")
                first_id = datasets[0]["id"] if isinstance(datasets[0], dict) else datasets[0].id
                df = DatasetRegistry.load_dataframe(first_id)

        target_column = req.target_column
        if not target_column:
            suggested = suggest_target_columns(df)
            target_column = suggested[0] if suggested else df.columns[-1]

        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{target_column}' not found.")

        task_type = req.task_type or "classification"
        model_name = req.model_name or "random_forest"

        runner = BenchmarkRunner()
        result = runner.run_benchmark(df, target_column, task_type, model_name)
        _auditor_state["benchmark_result"] = result

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

        return _clean_dict({
            "status": "completed",
            "overall": result.to_dict(),
            "scenarios": scenarios,
        })
    except Exception as e:
        logger.error(f"Benchmark error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {type(e).__name__}: {str(e)}")


@router.get("/findings/export")
async def export_findings():
    """Export all findings from the latest pipeline audit as a CSV file."""
    audit_result = _auditor_state.get("audit_result")
    if not audit_result or not audit_result.findings:
        raise HTTPException(status_code=404, detail="No audit findings available. Run an audit first.")

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

    csv_bytes = output.getvalue().encode("utf-8")
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=pipeline_audit_findings.csv"},
    )

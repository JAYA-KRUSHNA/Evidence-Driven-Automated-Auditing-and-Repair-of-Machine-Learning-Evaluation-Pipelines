"""
Machine Learning REST API endpoints.
"""

import json
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from ..config import (
    CLASSIFICATION_ALGORITHMS,
    DEFAULT_CV_FOLDS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TEST_SIZE,
    MODELS_DIR,
    REGRESSION_ALGORITHMS,
)
from ..database.connection import get_session
from ..database.models import ComparisonRun
from ..dataset.registry import DatasetNotFoundError, DatasetRegistry
from ..experiment.tracker import ExperimentTracker
from ..ml.classification import ClassificationEngine
from ..ml.comparison import ComparisonEngine
from ..ml.prediction import PredictionEngine
from ..ml.regression import RegressionEngine
from ..ml.task_identifier import TaskIdentifier
from ..models.registry import ModelNotFoundError, ModelRegistry

router = APIRouter(prefix="/api/ml", tags=["Machine Learning"])

_VALID_TASK_TYPES = {"classification", "regression"}


class TrainModelRequest(BaseModel):
    dataset_id: str
    target_column: str
    algorithm: str
    task_type: Optional[str] = None
    test_size: float = DEFAULT_TEST_SIZE
    cv_folds: int = DEFAULT_CV_FOLDS
    random_seed: int = DEFAULT_RANDOM_SEED
    hyperparameters: Optional[dict[str, Any]] = None
    experiment_name: Optional[str] = None


class CompareRequest(BaseModel):
    dataset_id: str
    target_column: str
    task_type: Optional[str] = None
    algorithms: Optional[list[str]] = None
    ranking_metric: Optional[str] = None
    config: Optional[dict[str, Any]] = None


class PredictRequest(BaseModel):
    model_id: str
    input_data: dict[str, Any] | list[dict[str, Any]]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _assert_valid_task_type(task_type: str) -> None:
    """Raise 400 if task_type is not one of the supported values."""
    if task_type not in _VALID_TASK_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid task_type '{task_type}'. "
                f"Supported values: {sorted(_VALID_TASK_TYPES)}. "
                f"Use task_type='classification' or task_type='regression'."
            ),
        )


def _validate_task_type_compatibility(
    requested: str,
    detected: str,
    target_column: str,
    detection_info: dict,
) -> None:
    """
    Raise HTTP 400 if the user-requested task_type clearly contradicts the data.

    Rules:
    - detected='regression' + requested='classification'  → reject:
      creates thousands of classes, crashes sklearn StratifiedKFold.
    - detected='classification' + requested='regression'  → reject:
      MAPE divides by zero on 0/1 labels giving trillion-scale garbage.
    """
    if requested == detected:
        return  # No conflict

    nunique = detection_info.get("nunique", 0)
    dtype = detection_info.get("target_dtype", "")
    confidence = detection_info.get("confidence", "low")

    if requested == "classification" and detected == "regression":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Task type mismatch: column '{target_column}' has {nunique} unique continuous "
                f"values (dtype={dtype}) — this is a regression target, not classification. "
                f"Using classification would create {nunique} classes, most with a single sample, "
                f"causing sklearn StratifiedKFold to crash. "
                f"Auto-detected task: regression (confidence={confidence}). "
                f"Either use task_type='regression' or bin the target into discrete categories first."
            ),
        )

    if requested == "regression" and detected == "classification":
        # Only block when there is a strong classification signal
        if confidence == "high" or nunique <= 20:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Task type mismatch: column '{target_column}' has only {nunique} unique "
                    f"values (dtype={dtype}) — this looks like a classification target. "
                    f"Running regression on class labels (e.g. 0/1) produces meaningless metrics: "
                    f"MAPE divides by zero giving infinite values, and R\u00b2 has no practical meaning. "
                    f"Auto-detected task: classification (confidence={confidence}). "
                    f"Use task_type='classification' and a classification algorithm instead."
                ),
            )


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/algorithms", status_code=status.HTTP_200_OK)
def list_algorithms():
    """List all supported classification and regression algorithms."""
    return {
        "classification": sorted(CLASSIFICATION_ALGORITHMS),
        "regression": sorted(REGRESSION_ALGORITHMS),
    }


@router.get("/identify-task/{dataset_id}", status_code=status.HTTP_200_OK)
@router.post("/identify-task/{dataset_id}", status_code=status.HTTP_200_OK)
def identify_task(dataset_id: str, target_column: str = Query(...)):
    """Automatically detect whether the target variable indicates classification or regression."""
    if not target_column or not target_column.strip():
        raise HTTPException(
            status_code=400,
            detail="target_column query parameter is required and cannot be empty",
        )
    try:
        df = DatasetRegistry.load_dataframe(dataset_id)
        return TaskIdentifier.identify(df, target_column)
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train", status_code=status.HTTP_200_OK)
def train_model(req: TrainModelRequest):
    """Train an ML model, track experiment run, and persist model artifact."""
    try:
        df = DatasetRegistry.load_dataframe(req.dataset_id)

        # Validate task_type if explicitly supplied
        if req.task_type:
            _assert_valid_task_type(req.task_type)

        # Detect actual task type from data
        detected = TaskIdentifier.identify(df, req.target_column)
        detected_task = detected["task_type"]

        # Resolve final task_type
        task_type = req.task_type or detected_task

        # Reject clear task/data mismatches before training
        if req.task_type:
            _validate_task_type_compatibility(
                req.task_type, detected_task, req.target_column, detected
            )

        # Validate the algorithm belongs to the chosen task
        if task_type == "classification" and req.algorithm not in CLASSIFICATION_ALGORITHMS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown classification algorithm '{req.algorithm}'. "
                    f"Available: {sorted(CLASSIFICATION_ALGORITHMS)}"
                ),
            )
        elif task_type == "regression" and req.algorithm not in REGRESSION_ALGORITHMS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown regression algorithm '{req.algorithm}'. "
                    f"Available: {sorted(REGRESSION_ALGORITHMS)}"
                ),
            )

        # Create experiment record
        exp = ExperimentTracker.create(
            dataset_id=req.dataset_id,
            task_type=task_type,
            target_column=req.target_column,
            algorithm=req.algorithm,
            hyperparameters=req.hyperparameters or {},
            test_size=req.test_size,
            cv_folds=req.cv_folds,
            random_seed=req.random_seed,
            name=req.experiment_name,
        )

        ExperimentTracker.update_status(exp["id"], "running")

        train_cfg = {
            "test_size": req.test_size,
            "cv_folds": req.cv_folds,
            "random_seed": req.random_seed,
            "hyperparameters": req.hyperparameters or {},
        }

        if task_type == "classification":
            train_result = ClassificationEngine.train(
                df=df,
                target_column=req.target_column,
                algorithm=req.algorithm,
                config=train_cfg,
            )
        else:
            train_result = RegressionEngine.train(
                df=df,
                target_column=req.target_column,
                algorithm=req.algorithm,
                config=train_cfg,
            )

        if train_result.error:
            ExperimentTracker.update_status(exp["id"], "failed", error_message=train_result.error)
            raise HTTPException(status_code=422, detail=f"Training failed: {train_result.error}")

        completed_exp = ExperimentTracker.save_results(
            experiment_id=exp["id"],
            training_result=train_result,
            is_best=True,
        )

        full_exp = ExperimentTracker.get(exp["id"])

        return {
            "status": "success",
            "experiment": completed_exp,
            "models": full_exp.get("models", []),
            "evaluations": full_exp.get("evaluations", []),
        }

    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset_id}' not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", status_code=status.HTTP_200_OK)
def compare_models(req: CompareRequest):
    """Train multiple algorithms side-by-side, rank by performance, and record comparison run."""
    session = get_session()
    try:
        df = DatasetRegistry.load_dataframe(req.dataset_id)

        # Validate task_type if explicitly supplied
        if req.task_type:
            _assert_valid_task_type(req.task_type)

        detected = TaskIdentifier.identify(df, req.target_column)
        detected_task = detected["task_type"]

        task_type = req.task_type or detected_task

        if req.task_type:
            _validate_task_type_compatibility(
                req.task_type, detected_task, req.target_column, detected
            )

        res = ComparisonEngine.compare(
            df=df,
            target_column=req.target_column,
            task_type=task_type,
            algorithms=req.algorithms,
            config=req.config,
        )

        comp_run = ComparisonRun(
            dataset_id=req.dataset_id,
            task_type=task_type,
            target_column=req.target_column,
            algorithms=res.algorithms_run,
            ranking_metric=res.ranking_metric,
            results=res.results,
            best_algorithm=res.best_algorithm,
        )
        session.add(comp_run)
        session.commit()

        return {
            "id": comp_run.id,
            "dataset_id": req.dataset_id,
            "task_type": res.task_type,
            "target_column": res.target_column,
            "algorithms_run": res.algorithms_run,
            "ranking_metric": res.ranking_metric,
            "best_algorithm": res.best_algorithm,
            "best_metrics": res.best_metrics,
            "results": res.results,
            "errors": res.errors,
            "created_at": comp_run.created_at.isoformat() if comp_run.created_at else None,
        }

    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset_id}' not found")
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@router.post("/predict", status_code=status.HTTP_200_OK)
def predict(req: PredictRequest):
    """Run inference against a saved model. Accepts a single dict or a list of dicts for batch prediction."""
    try:
        loaded = ModelRegistry.load(req.model_id)
        model = loaded["model"]
        meta = loaded["metadata"]
        label_encoder = loaded.get("label_encoder")

        task_type = meta.get("task_type", "classification")

        # Resolve feature columns: check metadata dict, then disk JSON, then model attribute
        feature_columns = meta.get("hyperparameters", {}).get("feature_columns")
        if not feature_columns:
            meta_json = MODELS_DIR / req.model_id / "metadata.json"
            if meta_json.exists():
                with open(meta_json, "r", encoding="utf-8") as f:
                    disk_meta = json.load(f)
                    feature_columns = disk_meta.get("feature_columns", [])

        if not feature_columns and hasattr(model, "feature_names_in_"):
            feature_columns = list(model.feature_names_in_)

        feature_columns = feature_columns or []

        # Route: single dict → predict, list of dicts → batch_predict
        if isinstance(req.input_data, list):
            if not req.input_data:
                raise HTTPException(status_code=400, detail="input_data list is empty")
            pred_result = PredictionEngine.batch_predict(
                model=model,
                df=__import__("pandas").DataFrame(req.input_data),
                feature_columns=feature_columns,
                label_encoder=label_encoder,
                task_type=task_type,
            )
        else:
            pred_result = PredictionEngine.predict(
                model=model,
                input_data=req.input_data,
                feature_columns=feature_columns,
                label_encoder=label_encoder,
                task_type=task_type,
            )

            # Log single prediction to experiment if experiment_id exists
            exp_id = meta.get("experiment_id")
            if exp_id:
                try:
                    ExperimentTracker.log_prediction(
                        experiment_id=exp_id,
                        input_data=req.input_data,
                        prediction=pred_result.get("prediction"),
                        model_id=req.model_id,
                        confidence=pred_result.get("confidence"),
                    )
                except Exception:
                    pass  # Logging failure must not break prediction response

        return pred_result

    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not found")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

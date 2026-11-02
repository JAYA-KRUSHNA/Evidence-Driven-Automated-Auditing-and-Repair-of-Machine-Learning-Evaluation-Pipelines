"""
Experiment Tracker — tracking training runs, hyperparameters, metrics, and models.

Public API
----------
    ExperimentTracker.create(...) -> dict
    ExperimentTracker.update_status(exp_id, status, ...) -> dict
    ExperimentTracker.save_results(exp_id, training_result, ...) -> dict
    ExperimentTracker.log_prediction(exp_id, ...) -> dict
    ExperimentTracker.get(exp_id) -> dict
    ExperimentTracker.list_all(...) -> list[dict]
    ExperimentTracker.compare(experiment_ids) -> dict
    ExperimentTracker.delete(exp_id) -> bool
"""

import logging
import shutil
from datetime import datetime, timezone
from typing import Any, Optional

from ..config import DEFAULT_CV_FOLDS, DEFAULT_RANDOM_SEED, DEFAULT_TEST_SIZE, MODELS_DIR
from ..database.connection import get_session
from ..database.models import Dataset, Experiment, EvaluationResult, Prediction
from ..models.registry import ModelRegistry

logger = logging.getLogger(__name__)


class ExperimentNotFoundError(Exception):
    """Raised when an experiment ID does not exist."""


class ExperimentTracker:
    """Manages machine learning experiments, evaluations, and history."""

    @classmethod
    def create(
        cls,
        dataset_id: str,
        task_type: str,
        target_column: str,
        algorithm: str,
        feature_columns: Optional[list[str]] = None,
        preprocessing_config: Optional[dict[str, Any]] = None,
        hyperparameters: Optional[dict[str, Any]] = None,
        test_size: float = DEFAULT_TEST_SIZE,
        cv_folds: int = DEFAULT_CV_FOLDS,
        random_seed: int = DEFAULT_RANDOM_SEED,
        name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a new experiment record in 'pending' status."""
        session = get_session()
        try:
            # Check dataset exists
            ds = session.query(Dataset).filter_by(id=dataset_id).first()
            if not ds:
                raise ValueError(f"Dataset '{dataset_id}' not found.")

            exp_name = name or f"{algorithm.replace('_', ' ').title()} on {ds.name}"

            exp = Experiment(
                dataset_id=dataset_id,
                name=exp_name,
                task_type=task_type,
                target_column=target_column,
                feature_columns=feature_columns or [],
                preprocessing_config=preprocessing_config or {},
                algorithm=algorithm,
                hyperparameters=hyperparameters or {},
                test_size=float(test_size),
                cv_folds=int(cv_folds),
                random_seed=int(random_seed),
                status="pending",
            )
            session.add(exp)
            session.commit()
            logger.info("Created experiment %s (%s)", exp.id, exp_name)
            return cls._to_dict(exp)
        except Exception:
            session.rollback()
            logger.exception("Failed to create experiment")
            raise
        finally:
            session.close()

    @classmethod
    def update_status(
        cls,
        experiment_id: str,
        status: str,
        error_message: str = "",
        training_time_sec: float = 0.0,
    ) -> dict[str, Any]:
        """Update status ('running', 'completed', 'failed') and error message."""
        session = get_session()
        try:
            exp = session.query(Experiment).filter_by(id=experiment_id).first()
            if not exp:
                raise ExperimentNotFoundError(f"Experiment '{experiment_id}' not found.")

            exp.status = status
            if error_message:
                exp.error_message = error_message
            if training_time_sec > 0:
                exp.training_time_sec = float(training_time_sec)
            if status in ("completed", "failed"):
                exp.completed_at = datetime.now(timezone.utc)

            session.commit()
            return cls._to_dict(exp)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @classmethod
    def save_results(
        cls,
        experiment_id: str,
        training_result: Any,
        is_best: bool = False,
        preprocessing_pipeline: Optional[Any] = None,
    ) -> dict[str, Any]:
        """
        Record training outcome for an experiment.

        Saves:
        - TrainedModel in ModelRegistry
        - EvaluationResult records for each metric
        - Updates Experiment status to 'completed'
        """
        session = get_session()
        try:
            exp = session.query(Experiment).filter_by(id=experiment_id).first()
            if not exp:
                raise ExperimentNotFoundError(f"Experiment '{experiment_id}' not found.")

            metrics = getattr(training_result, "metrics", {}) or {}
            training_time = float(getattr(training_result, "training_time_sec", 0.0))
            algorithm = getattr(training_result, "algorithm", exp.algorithm)
            task_type = getattr(training_result, "task_type", exp.task_type)
            model_obj = getattr(training_result, "model", None)
            label_encoder = getattr(training_result, "label_encoder", None)
            feature_cols = getattr(training_result, "feature_columns", exp.feature_columns)
            target_col = getattr(training_result, "target_column", exp.target_column)

            # If error in training_result
            error = getattr(training_result, "error", "")
            if error:
                exp.status = "failed"
                exp.error_message = error
                session.commit()
                return cls._to_dict(exp)

            # 1. Save model via ModelRegistry if model object exists
            model_record = None
            if model_obj is not None:
                model_record = ModelRegistry.save(
                    model=model_obj,
                    algorithm=algorithm,
                    task_type=task_type,
                    experiment_id=experiment_id,
                    dataset_id=exp.dataset_id,
                    model_name=f"{exp.name} Model",
                    metrics=metrics,
                    hyperparameters=exp.hyperparameters,
                    training_time_sec=training_time,
                    label_encoder=label_encoder,
                    preprocessing_pipeline=preprocessing_pipeline,
                    is_best=is_best,
                    feature_columns=feature_cols,
                    target_column=target_col,
                )

            # 2. Record EvaluationResult entries for scalar metrics only.
            # Non-scalar metrics (confusion_matrix, per_class, class_labels) are
            # stored in the model's metrics JSON column and returned via models[].metrics.
            _NON_SCALAR = {"confusion_matrix", "per_class", "class_labels", "mape_note",
                           "feature_importances", "feature_coef"}
            model_id = model_record["id"] if model_record else None
            for metric_name, metric_value in metrics.items():
                if metric_name in _NON_SCALAR:
                    continue  # stored in JSON column, not as scalar EvaluationResult rows
                if isinstance(metric_value, (int, float)) and metric_value is not None:
                    eval_rec = EvaluationResult(
                        experiment_id=experiment_id,
                        model_id=model_id,
                        metric_name=metric_name,
                        metric_value=float(metric_value),
                        split="test",
                    )
                    session.add(eval_rec)

            # 3. Add CV scores if present
            cv_scores = getattr(training_result, "cv_scores", [])
            if cv_scores:
                for idx, score in enumerate(cv_scores):
                    eval_rec = EvaluationResult(
                        experiment_id=experiment_id,
                        model_id=model_id,
                        metric_name=f"cv_fold_{idx+1}",
                        metric_value=float(score),
                        split="cv",
                    )
                    session.add(eval_rec)

            exp.status = "completed"
            exp.training_time_sec = training_time
            exp.completed_at = datetime.now(timezone.utc)
            session.commit()

            return cls._to_dict(exp)
        except Exception:
            session.rollback()
            logger.exception("Failed to save experiment results")
            raise
        finally:
            session.close()

    @classmethod
    def log_prediction(
        cls,
        experiment_id: str,
        input_data: dict[str, Any] | list[Any],
        prediction: Any,
        model_id: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> dict[str, Any]:
        """Record an inference event linked to an experiment."""
        session = get_session()
        try:
            exp = session.query(Experiment).filter_by(id=experiment_id).first()
            if not exp:
                raise ExperimentNotFoundError(f"Experiment '{experiment_id}' not found.")

            pred_record = Prediction(
                experiment_id=experiment_id,
                model_id=model_id,
                input_data=input_data,
                prediction=prediction,
                confidence=float(confidence) if confidence is not None else None,
            )
            session.add(pred_record)
            session.commit()

            return {
                "id": pred_record.id,
                "experiment_id": experiment_id,
                "model_id": model_id,
                "prediction": prediction,
                "confidence": confidence,
                "created_at": pred_record.created_at.isoformat() if pred_record.created_at else None,
            }
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @classmethod
    def get(cls, experiment_id: str) -> dict[str, Any]:
        """Retrieve full details of an experiment including models and evaluations."""
        session = get_session()
        try:
            exp = session.query(Experiment).filter_by(id=experiment_id).first()
            if not exp:
                raise ExperimentNotFoundError(f"Experiment '{experiment_id}' not found.")
            return cls._to_dict(exp, include_relations=True)
        finally:
            session.close()

    @classmethod
    def list_all(
        cls,
        dataset_id: Optional[str] = None,
        task_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """List experiments with optional filters."""
        session = get_session()
        try:
            query = session.query(Experiment)
            if dataset_id:
                query = query.filter_by(dataset_id=dataset_id)
            if task_type:
                query = query.filter_by(task_type=task_type)
            if status:
                query = query.filter_by(status=status)

            query = query.order_by(Experiment.created_at.desc())
            return [cls._to_dict(e) for e in query.all()]
        finally:
            session.close()

    @classmethod
    def compare(cls, experiment_ids: list[str]) -> dict[str, Any]:
        """Compare multiple experiments side by side."""
        session = get_session()
        try:
            exps = (
                session.query(Experiment)
                .filter(Experiment.id.in_(experiment_ids))
                .all()
            )
            if not exps:
                return {"experiments": [], "metrics_comparison": {}}

            results = []
            all_metric_names = set()

            for exp in exps:
                details = cls._to_dict(exp, include_relations=True)
                metrics = {}
                for ev in details.get("evaluations", []):
                    if ev["split"] == "test":
                        metrics[ev["metric_name"]] = ev["metric_value"]
                        all_metric_names.add(ev["metric_name"])

                results.append({
                    "id": exp.id,
                    "name": exp.name,
                    "algorithm": exp.algorithm,
                    "task_type": exp.task_type,
                    "status": exp.status,
                    "training_time_sec": exp.training_time_sec,
                    "metrics": metrics,
                    "hyperparameters": exp.hyperparameters or {},
                })

            return {
                "experiments": results,
                "compared_metrics": sorted(list(all_metric_names)),
                "total_compared": len(results),
            }
        finally:
            session.close()

    @classmethod
    def delete(cls, experiment_id: str) -> bool:
        """Delete an experiment and its cascading relations."""
        session = get_session()
        try:
            exp = session.query(Experiment).filter_by(id=experiment_id).first()
            if not exp:
                raise ExperimentNotFoundError(f"Experiment '{experiment_id}' not found.")

            # Remove model files on disk
            for model in exp.models:
                model_dir = MODELS_DIR / model.id
                if model_dir.exists():
                    shutil.rmtree(model_dir, ignore_errors=True)

            session.delete(exp)
            session.commit()
            logger.info("Deleted experiment %s", experiment_id)
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _to_dict(exp: Experiment, include_relations: bool = False) -> dict[str, Any]:
        data = {
            "id": exp.id,
            "dataset_id": exp.dataset_id,
            "name": exp.name,
            "task_type": exp.task_type,
            "target_column": exp.target_column,
            "feature_columns": exp.feature_columns or [],
            "preprocessing_config": exp.preprocessing_config or {},
            "algorithm": exp.algorithm,
            "hyperparameters": exp.hyperparameters or {},
            "test_size": exp.test_size,
            "cv_folds": exp.cv_folds,
            "random_seed": exp.random_seed,
            "training_time_sec": exp.training_time_sec,
            "status": exp.status,
            "error_message": exp.error_message or "",
            "created_at": exp.created_at.isoformat() if exp.created_at else None,
            "completed_at": exp.completed_at.isoformat() if exp.completed_at else None,
        }

        if include_relations:
            data["models"] = [
                {
                    "id": m.id,
                    "model_name": m.model_name,
                    "algorithm": m.algorithm,
                    "metrics": m.metrics or {},
                    "is_best": m.is_best,
                }
                for m in getattr(exp, "models", [])
            ]
            data["evaluations"] = [
                {
                    "id": ev.id,
                    "metric_name": ev.metric_name,
                    "metric_value": ev.metric_value,
                    "split": ev.split,
                }
                for ev in getattr(exp, "evaluations", [])
            ]
            data["predictions_count"] = len(getattr(exp, "predictions", []))

        return data

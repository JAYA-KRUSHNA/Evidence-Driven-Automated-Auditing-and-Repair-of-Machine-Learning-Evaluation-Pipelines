"""
Model Registry — persistent storage, retrieval, and export of trained ML models.

Supports:
- Model serialization via Joblib
- Persistent metadata in SQLite (`trained_models` and `model_files` tables)
- Associated artifact storage (pipeline, label_encoder)
- Listing, filtering, deletion, and export (as joblib or zip bundle)

Public API
----------
    ModelRegistry.save(...) -> dict
    ModelRegistry.load(model_id) -> dict
    ModelRegistry.get(model_id) -> dict
    ModelRegistry.list_all(...) -> list[dict]
    ModelRegistry.delete(model_id) -> bool
    ModelRegistry.export(model_id, ...) -> str
    ModelRegistry.set_best(model_id) -> None
"""

import json
import logging
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib

from ..config import MODELS_DIR
from ..database.connection import get_session
from ..database.models import Dataset, Experiment, ModelFile, TrainedModel

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """Raised when a model ID cannot be found in the registry."""


class ModelRegistry:
    """Manages trained model artifacts and database tracking."""

    @staticmethod
    def _ensure_experiment(session, experiment_id: Optional[str] = None, dataset_id: Optional[str] = None, task_type: str = "classification") -> str:
        """Helper to ensure a valid experiment_id exists for foreign key constraints."""
        if experiment_id:
            exp = session.query(Experiment).filter_by(id=experiment_id).first()
            if exp:
                return experiment_id

        # Need to create or find an experiment
        # Ensure a dataset exists first
        if dataset_id:
            ds = session.query(Dataset).filter_by(id=dataset_id).first()
        else:
            ds = session.query(Dataset).first()

        if not ds:
            # Create a default placeholder dataset
            ds = Dataset(
                name="Default Dataset",
                original_name="placeholder.csv",
                file_path=str(MODELS_DIR / "placeholder.csv"),
                file_format="csv",
                source="auto_generated",
                num_rows=100,
                num_columns=5,
            )
            session.add(ds)
            session.flush()

        new_exp = Experiment(
            id=experiment_id if experiment_id else None,
            dataset_id=ds.id,
            name="Auto Experiment",
            task_type=task_type,
            target_column="target",
            algorithm="auto",
            status="completed",
        )
        session.add(new_exp)
        session.flush()
        return new_exp.id

    @classmethod
    def save(
        cls,
        model: Any,
        algorithm: str,
        task_type: str,
        experiment_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        model_name: Optional[str] = None,
        metrics: Optional[dict[str, Any]] = None,
        hyperparameters: Optional[dict[str, Any]] = None,
        training_time_sec: float = 0.0,
        label_encoder: Optional[Any] = None,
        preprocessing_pipeline: Optional[Any] = None,
        is_best: bool = False,
        version: int = 1,
        feature_columns: Optional[list[str]] = None,
        target_column: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Save a trained model and any artifacts to disk and register in DB.

        Returns
        -------
        dict with complete model metadata and file paths.
        """
        session = get_session()
        try:
            valid_exp_id = cls._ensure_experiment(session, experiment_id, dataset_id, task_type)

            name = model_name or f"{algorithm.replace('_', ' ').title()} Model"

            # Check if there are other models in this experiment to set version
            if version == 1:
                existing_count = (
                    session.query(TrainedModel)
                    .filter_by(experiment_id=valid_exp_id, algorithm=algorithm)
                    .count()
                )
                version = existing_count + 1

            trained_model = TrainedModel(
                experiment_id=valid_exp_id,
                model_name=name,
                algorithm=algorithm,
                task_type=task_type,
                hyperparameters=hyperparameters or {},
                metrics=metrics or {},
                training_time_sec=float(training_time_sec),
                version=version,
                is_best=is_best,
            )
            session.add(trained_model)
            session.flush()  # generates trained_model.id

            model_id = trained_model.id
            model_dir = MODELS_DIR / model_id
            model_dir.mkdir(parents=True, exist_ok=True)

            # 1. Save main model
            model_path = model_dir / "model.joblib"
            joblib.dump(model, model_path)
            model_file_rec = ModelFile(
                model_id=model_id,
                file_type="model",
                file_path=str(model_path),
                file_size=model_path.stat().st_size,
            )
            session.add(model_file_rec)

            # 2. Save label encoder if present
            if label_encoder is not None:
                le_path = model_dir / "label_encoder.joblib"
                joblib.dump(label_encoder, le_path)
                le_file_rec = ModelFile(
                    model_id=model_id,
                    file_type="label_encoder",
                    file_path=str(le_path),
                    file_size=le_path.stat().st_size,
                )
                session.add(le_file_rec)

            # 3. Save preprocessing pipeline if present
            pipeline_path_str = ""
            if preprocessing_pipeline is not None:
                if isinstance(preprocessing_pipeline, (str, Path)):
                    pipeline_path_str = str(preprocessing_pipeline)
                else:
                    pipe_path = model_dir / "pipeline.joblib"
                    joblib.dump(preprocessing_pipeline, pipe_path)
                    pipe_file_rec = ModelFile(
                        model_id=model_id,
                        file_type="pipeline",
                        file_path=str(pipe_path),
                        file_size=pipe_path.stat().st_size,
                    )
                    session.add(pipe_file_rec)
                    pipeline_path_str = str(pipe_path)
                trained_model.preprocessing_pipeline_path = pipeline_path_str

            # 4. Save metadata.json locally for easy inspection
            meta = {
                "id": model_id,
                "experiment_id": valid_exp_id,
                "model_name": name,
                "algorithm": algorithm,
                "task_type": task_type,
                "metrics": metrics or {},
                "hyperparameters": hyperparameters or {},
                "feature_columns": feature_columns or [],
                "target_column": target_column or "",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            if is_best:
                # Unset previous is_best for this experiment
                session.query(TrainedModel).filter(
                    TrainedModel.experiment_id == valid_exp_id,
                    TrainedModel.id != model_id,
                ).update({"is_best": False})

            session.commit()
            logger.info("Saved model %s (%s) to %s", model_id, name, model_dir)
            return cls._to_dict(trained_model)

        except Exception:
            session.rollback()
            logger.exception("Failed to save model")
            raise
        finally:
            session.close()

    @classmethod
    def load(cls, model_id: str) -> dict[str, Any]:
        """
        Load a trained model and all attached components from disk.

        Returns
        -------
        dict with:
            'model': fitted estimator
            'metadata': dict of metadata
            'label_encoder': LabelEncoder or None
            'pipeline': Pipeline or None
        """
        metadata = cls.get(model_id)
        model_dir = MODELS_DIR / model_id

        model_file = model_dir / "model.joblib"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found at {model_file}")

        loaded_model = joblib.load(model_file)

        label_encoder = None
        le_file = model_dir / "label_encoder.joblib"
        if le_file.exists():
            label_encoder = joblib.load(le_file)

        pipeline = None
        pipe_file = model_dir / "pipeline.joblib"
        if pipe_file.exists():
            pipeline = joblib.load(pipe_file)

        return {
            "model": loaded_model,
            "metadata": metadata,
            "label_encoder": label_encoder,
            "pipeline": pipeline,
        }

    @classmethod
    def get(cls, model_id: str) -> dict[str, Any]:
        """Retrieve model metadata by ID."""
        session = get_session()
        try:
            m = session.query(TrainedModel).filter_by(id=model_id).first()
            if not m:
                raise ModelNotFoundError(f"Model '{model_id}' does not exist.")
            return cls._to_dict(m)
        finally:
            session.close()

    @classmethod
    def list_all(
        cls,
        dataset_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        task_type: Optional[str] = None,
        algorithm: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """List all models matching the optional filters."""
        session = get_session()
        try:
            query = session.query(TrainedModel)

            if dataset_id:
                query = query.join(Experiment, TrainedModel.experiment_id == Experiment.id).filter(
                    Experiment.dataset_id == dataset_id
                )

            if experiment_id:
                query = query.filter(TrainedModel.experiment_id == experiment_id)

            if task_type:
                query = query.filter(TrainedModel.task_type == task_type)

            if algorithm:
                query = query.filter(TrainedModel.algorithm == algorithm)

            query = query.order_by(TrainedModel.created_at.desc())
            models = query.all()
            return [cls._to_dict(m) for m in models]
        finally:
            session.close()

    @classmethod
    def delete(cls, model_id: str) -> bool:
        """Delete a model record from DB and remove its disk artifacts."""
        session = get_session()
        try:
            m = session.query(TrainedModel).filter_by(id=model_id).first()
            if not m:
                raise ModelNotFoundError(f"Model '{model_id}' does not exist.")

            session.delete(m)
            session.commit()

            # Remove disk directory
            model_dir = MODELS_DIR / model_id
            if model_dir.exists():
                shutil.rmtree(model_dir, ignore_errors=True)

            logger.info("Deleted model %s", model_id)
            return True
        except Exception:
            session.rollback()
            logger.exception("Failed to delete model %s: %s", model_id)
            raise
        finally:
            session.close()

    @classmethod
    def export(
        cls,
        model_id: str,
        format: str = "joblib",
        target_path: Optional[str | Path] = None,
    ) -> str:
        """
        Export a model to a specified destination or format.
        format: 'joblib' (single model file) or 'zip' (bundle with metadata & artifacts)
        """
        cls.get(model_id)  # validates model exists — raises if not found
        model_dir = MODELS_DIR / model_id

        if format == "joblib":
            src = model_dir / "model.joblib"
            if not src.exists():
                raise FileNotFoundError(f"Model file missing: {src}")

            if target_path:
                dst = Path(target_path)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                return str(dst)
            return str(src)

        elif format == "zip":
            if target_path:
                dst_zip = Path(target_path)
            else:
                dst_zip = model_dir.parent / f"{model_id}_export.zip"

            dst_zip.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(dst_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in model_dir.rglob("*"):
                    if file_path.is_file():
                        zf.write(file_path, arcname=file_path.relative_to(model_dir))
            return str(dst_zip)
        else:
            raise ValueError(f"Unsupported export format '{format}'. Use 'joblib' or 'zip'.")

    @classmethod
    def set_best(cls, model_id: str) -> None:
        """Mark a specific model as the best in its experiment."""
        session = get_session()
        try:
            m = session.query(TrainedModel).filter_by(id=model_id).first()
            if not m:
                raise ModelNotFoundError(f"Model '{model_id}' does not exist.")

            # Unset others in same experiment
            session.query(TrainedModel).filter(
                TrainedModel.experiment_id == m.experiment_id,
                TrainedModel.id != model_id,
            ).update({"is_best": False})

            m.is_best = True
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _to_dict(model: TrainedModel) -> dict[str, Any]:
        """Convert a TrainedModel ORM row to a JSON-serializable dictionary."""
        files = []
        for mf in getattr(model, "model_files", []):
            files.append({
                "id": mf.id,
                "file_type": mf.file_type,
                "file_path": mf.file_path,
                "file_size": mf.file_size,
                "created_at": mf.created_at.isoformat() if mf.created_at else None,
            })

        return {
            "id": model.id,
            "experiment_id": model.experiment_id,
            "model_name": model.model_name,
            "algorithm": model.algorithm,
            "task_type": model.task_type,
            "hyperparameters": model.hyperparameters or {},
            "preprocessing_pipeline_path": model.preprocessing_pipeline_path,
            "metrics": model.metrics or {},
            "training_time_sec": model.training_time_sec,
            "version": model.version,
            "is_best": model.is_best,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "files": files,
        }

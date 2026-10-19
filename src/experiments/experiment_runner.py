"""
Experiment Runner — Orchestrates the full experiment lifecycle.

Coordinates:
1. Data loading and profiling
2. Feature/target separation
3. Splitting
4. Training
5. Evaluation
6. Audit
7. Storage of results

All operations use real data and real model training.
"""

import logging
import time
import json
from typing import Optional

import pandas as pd
import numpy as np

from src.config import (
    DEFAULT_RANDOM_SEED, DEFAULT_TEST_SIZE, DEFAULT_CV_FOLDS,
    MAX_TRAINING_ROWS, get_software_versions
)
from src.data.dataset_profiler import DatasetProfiler
from src.experiments.split_manager import SplitManager
from src.experiments.pipeline_builder import PipelineBuilder
from src.experiments.trainer import Trainer, TrainingResult
from src.database.models import get_session, init_db
from src.database.repository import (
    ExperimentRepository, MetricRepository, DatasetRepository
)

logger = logging.getLogger(__name__)


class ExperimentConfig:
    """Configuration for an experiment run."""

    def __init__(
        self,
        dataset_id: str,
        target_column: str,
        task_type: str = "classification",
        model_names: list[str] = None,
        test_size: float = DEFAULT_TEST_SIZE,
        cv_folds: int = DEFAULT_CV_FOLDS,
        random_seed: int = DEFAULT_RANDOM_SEED,
        split_strategy: str = "auto",
        experiment_name: str = "",
    ):
        self.dataset_id = dataset_id
        self.target_column = target_column
        self.task_type = task_type
        self.model_names = model_names or ["random_forest"]
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_seed = random_seed
        self.split_strategy = split_strategy
        self.experiment_name = experiment_name or f"Experiment_{int(time.time())}"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class ExperimentResult:
    """Container for full experiment results."""

    def __init__(self):
        self.experiment_id: str = ""
        self.config: Optional[ExperimentConfig] = None
        self.profile: Optional[dict] = None
        self.target_analysis: Optional[dict] = None
        self.training_results: dict[str, TrainingResult] = {}
        self.split_info: dict = {}
        self.total_time: float = 0.0
        self.status: str = "pending"
        self.error: Optional[str] = None
        self.software_versions: dict = {}


class ExperimentRunner:
    """
    Runs complete ML experiments with real training.

    This is the main orchestrator that coordinates all components.
    """

    def __init__(self):
        self.profiler = DatasetProfiler()

    def run(self, df: pd.DataFrame,
            config: ExperimentConfig,
            progress_callback=None) -> ExperimentResult:
        """
        Execute a complete experiment.

        Args:
            df: The full dataset DataFrame
            config: Experiment configuration
            progress_callback: Optional callable(step, total, message)

        Returns:
            ExperimentResult with all metrics and training results
        """
        result = ExperimentResult()
        result.config = config
        result.software_versions = get_software_versions()
        start_time = time.time()

        def report_progress(step, total, msg):
            if progress_callback:
                progress_callback(step, total, msg)
            logger.info(f"[{step}/{total}] {msg}")

        try:
            total_steps = 6

            # Step 1: Validate and profile
            report_progress(1, total_steps, "Profiling dataset...")
            self._validate_inputs(df, config)
            profile = self.profiler.profile(df, config.target_column)
            result.profile = profile.to_dict()
            result.target_analysis = self.profiler.get_target_analysis(
                df, config.target_column
            )

            # Step 2: Prepare features and target
            report_progress(2, total_steps, "Preparing features and target...")
            X, y = self._prepare_data(df, config)

            # Step 3: Split data
            report_progress(3, total_steps, "Splitting data...")
            X_train, X_test, y_train, y_test = self._split_data(X, y, config)
            result.split_info = {
                "train_size": len(X_train),
                "test_size": len(X_test),
                "strategy": config.split_strategy,
                "test_size_pct": config.test_size,
                "random_seed": config.random_seed,
            }

            # Check for overlap (important for audit later)
            overlap_info = SplitManager.check_split_overlap(X_train, X_test)
            result.split_info["overlap"] = overlap_info

            # Step 4: Store experiment in database
            report_progress(4, total_steps, "Storing experiment metadata...")
            result.experiment_id = self._store_experiment(config, result)

            # Step 5: Train models
            report_progress(5, total_steps, "Training models...")
            trainer = Trainer(config.random_seed)

            for i, model_name in enumerate(config.model_names):
                report_progress(
                    5, total_steps,
                    f"Training model {i+1}/{len(config.model_names)}: {model_name}"
                )
                try:
                    training_result = trainer.train_and_evaluate(
                        X_train, X_test, y_train, y_test,
                        model_name=model_name,
                        task_type=config.task_type,
                        n_cv_folds=config.cv_folds,
                    )
                    result.training_results[model_name] = training_result

                    # Store metrics
                    self._store_metrics(
                        result.experiment_id, model_name, training_result
                    )
                except Exception as e:
                    logger.error(f"Training failed for {model_name}: {e}")
                    # Continue with other models
                    result.training_results[model_name] = TrainingResult()
                    result.training_results[model_name].model_name = model_name
                    result.training_results[model_name].test_metrics = {
                        "error": str(e)
                    }

            # Step 6: Finalize
            report_progress(6, total_steps, "Finalizing experiment...")
            result.total_time = round(time.time() - start_time, 2)
            result.status = "completed"

            # Update experiment status in DB
            self._update_experiment_status(
                result.experiment_id, "completed",
                duration_seconds=result.total_time
            )

        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            result.total_time = round(time.time() - start_time, 2)
            logger.error(f"Experiment failed: {e}", exc_info=True)

            if result.experiment_id:
                self._update_experiment_status(
                    result.experiment_id, "failed",
                    error_message=str(e),
                    duration_seconds=result.total_time
                )

        return result

    def _validate_inputs(self, df: pd.DataFrame, config: ExperimentConfig):
        """Validate experiment inputs."""
        if df.empty:
            raise ValueError("Dataset is empty.")

        if config.target_column not in df.columns:
            raise ValueError(
                f"Target column '{config.target_column}' not found. "
                f"Available: {', '.join(df.columns[:20])}"
            )

        if len(df) < 10:
            raise ValueError(
                f"Dataset too small ({len(df)} rows). "
                "Need at least 10 rows for meaningful training."
            )

        # Validate task type matches target
        target_series = df[config.target_column].dropna()
        unique_count = target_series.nunique()

        if config.task_type == "classification" and unique_count > 100:
            logger.warning(
                f"Target has {unique_count} unique values for classification. "
                "Consider using regression instead."
            )

    def _prepare_data(self, df: pd.DataFrame,
                      config: ExperimentConfig):
        """Separate features from target, handle basic cleaning."""
        # Drop rows where target is missing
        df_clean = df.dropna(subset=[config.target_column]).copy()

        if len(df_clean) < len(df):
            dropped = len(df) - len(df_clean)
            logger.info(f"Dropped {dropped} rows with missing target values.")

        # Separate X and y
        y = df_clean[config.target_column].copy()
        X = df_clean.drop(columns=[config.target_column]).copy()

        # Encode target for classification if needed
        if config.task_type == "classification":
            y_encoded, _ = PipelineBuilder.encode_target(y, "classification")
            y = y_encoded

        # Sample if too large
        if len(X) > MAX_TRAINING_ROWS:
            logger.info(
                f"Dataset has {len(X)} rows. Sampling {MAX_TRAINING_ROWS}."
            )
            sample_idx = X.sample(
                n=MAX_TRAINING_ROWS, random_state=config.random_seed
            ).index
            X = X.loc[sample_idx]
            y = y.loc[sample_idx]

        # Remove columns that would cause issues
        # (all missing, all same value)
        cols_to_drop = []
        for col in X.columns:
            if X[col].isna().all():
                cols_to_drop.append(col)
            elif X[col].nunique() <= 1:
                cols_to_drop.append(col)

        if cols_to_drop:
            logger.info(f"Dropping {len(cols_to_drop)} unusable columns: {cols_to_drop}")
            X = X.drop(columns=cols_to_drop)

        if X.empty:
            raise ValueError("No usable feature columns remaining after cleaning.")

        return X, y

    def _split_data(self, X, y, config):
        """Split data using the configured strategy."""
        return SplitManager.train_test_split(
            X, y,
            test_size=config.test_size,
            random_seed=config.random_seed,
            stratify=(config.task_type == "classification"),
            task_type=config.task_type,
        )

    def _store_experiment(self, config: ExperimentConfig,
                          result: ExperimentResult) -> str:
        """Store experiment metadata in the database."""
        try:
            init_db()
            session = get_session()

            experiment = ExperimentRepository.create(
                session,
                dataset_id=config.dataset_id,
                name=config.experiment_name,
                task_type=config.task_type,
                target_column=config.target_column,
                model_config=json.dumps({
                    "models": config.model_names,
                }),
                split_config=json.dumps({
                    "strategy": config.split_strategy,
                    "test_size": config.test_size,
                    "cv_folds": config.cv_folds,
                }),
                random_seed=config.random_seed,
                software_versions=json.dumps(result.software_versions),
                status="running",
            )

            session.close()
            return experiment.id

        except Exception as e:
            logger.error(f"Failed to store experiment: {e}")
            return f"local_{int(time.time())}"

    def _store_metrics(self, experiment_id: str, model_name: str,
                       training_result: TrainingResult):
        """Store training metrics in the database."""
        try:
            session = get_session()

            # Store test metrics
            test_metrics = {
                k: v for k, v in training_result.test_metrics.items()
                if isinstance(v, (int, float)) and k != "confusion_matrix"
            }
            MetricRepository.create_bulk(
                session, experiment_id, test_metrics, split_type="test"
            )

            # Store train metrics
            train_metrics = {
                k: v for k, v in training_result.train_metrics.items()
                if isinstance(v, (int, float)) and k != "confusion_matrix"
            }
            MetricRepository.create_bulk(
                session, experiment_id, train_metrics, split_type="train"
            )

            # Store CV metrics
            cv_metrics = {
                k: v for k, v in training_result.cv_metrics.items()
                if isinstance(v, (int, float))
            }
            MetricRepository.create_bulk(
                session, experiment_id, cv_metrics, split_type="cv"
            )

            session.close()

        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")

    def _update_experiment_status(self, experiment_id: str,
                                   status: str, **kwargs):
        """Update experiment status in the database."""
        try:
            session = get_session()
            ExperimentRepository.update_status(
                session, experiment_id, status, **kwargs
            )
            session.close()
        except Exception as e:
            logger.error(f"Failed to update experiment status: {e}")

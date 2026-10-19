"""
Benchmark Runner & Metrics — Evaluates the auditor itself.

Runs the auditor against controlled corruption scenarios and
measures detection precision, recall, F1, and false positive rate.
"""

import logging
import json
import time
from typing import Optional

import pandas as pd
import numpy as np

from src.benchmark.corruption_generator import CorruptionGenerator, CorruptionScenario
from src.audit.audit_engine import AuditEngine, AuditResult
from src.experiments.experiment_runner import ExperimentRunner, ExperimentConfig
from src.experiments.split_manager import SplitManager
from src.experiments.pipeline_builder import PipelineBuilder
from src.database.models import get_session, init_db
from src.database.repository import BenchmarkRepository
from src.config import DEFAULT_RANDOM_SEED, AuditCategory

logger = logging.getLogger(__name__)


class BenchmarkResult:
    """Result of running the auditor against one scenario."""

    def __init__(self):
        self.scenario_name: str = ""
        self.corruption_type: str = ""
        self.ground_truth: list[str] = []
        self.detected_categories: list[str] = []
        self.true_positives: list[str] = []
        self.false_positives: list[str] = []
        self.false_negatives: list[str] = []
        self.precision: float = 0.0
        self.recall: float = 0.0
        self.f1: float = 0.0
        self.audit_findings_count: int = 0
        self.is_holdout: bool = False
        self.error: Optional[str] = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class AggregateBenchmarkResult:
    """Aggregate results across all scenarios."""

    def __init__(self):
        self.scenario_results: list[BenchmarkResult] = []
        self.overall_precision: float = 0.0
        self.overall_recall: float = 0.0
        self.overall_f1: float = 0.0
        self.overall_false_positive_rate: float = 0.0
        self.per_category: dict = {}
        self.total_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "overall_precision": round(self.overall_precision, 4),
            "overall_recall": round(self.overall_recall, 4),
            "overall_f1": round(self.overall_f1, 4),
            "overall_false_positive_rate": round(self.overall_false_positive_rate, 4),
            "per_category": self.per_category,
            "scenario_count": len(self.scenario_results),
            "total_time": round(self.total_time, 2),
        }


class BenchmarkRunner:
    """
    Evaluates the auditor's detection accuracy using controlled experiments.

    Generates corrupted datasets with known ground truth, runs the auditor,
    and measures precision/recall of detection.
    """

    def __init__(self, random_seed: int = DEFAULT_RANDOM_SEED):
        self.seed = random_seed
        self.generator = CorruptionGenerator(random_seed)
        self.audit_engine = AuditEngine()

    def run_benchmark(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str = "classification",
        model_name: str = "random_forest",
        progress_callback=None,
    ) -> AggregateBenchmarkResult:
        """
        Run the full benchmark suite.

        Args:
            df: Base dataset (real data)
            target_column: Target column
            task_type: classification or regression
            model_name: Model to use in experiments
            progress_callback: Optional callable(step, total, message)
        """
        start_time = time.time()
        aggregate = AggregateBenchmarkResult()

        # Generate scenarios
        scenarios = self.generator.generate_all_scenarios(
            df, target_column, task_type
        )

        total = len(scenarios)
        logger.info(f"Running benchmark with {total} scenarios...")

        for i, scenario in enumerate(scenarios):
            if progress_callback:
                progress_callback(
                    i + 1, total,
                    f"Scenario: {scenario.name}"
                )

            try:
                result = self._run_single_scenario(
                    scenario, target_column, task_type, model_name
                )
                aggregate.scenario_results.append(result)
            except Exception as e:
                logger.error(f"Scenario '{scenario.name}' failed: {e}")
                br = BenchmarkResult()
                br.scenario_name = scenario.name
                br.corruption_type = scenario.corruption_type
                br.ground_truth = scenario.ground_truth_issues
                br.error = str(e)
                aggregate.scenario_results.append(br)

        # Calculate aggregate metrics
        self._calculate_aggregate_metrics(aggregate)
        aggregate.total_time = round(time.time() - start_time, 2)

        # Store results
        self._store_results(aggregate, df, target_column)

        return aggregate

    def _run_single_scenario(
        self,
        scenario: CorruptionScenario,
        target_column: str,
        task_type: str,
        model_name: str,
    ) -> BenchmarkResult:
        """Run auditor against a single corruption scenario."""
        result = BenchmarkResult()
        result.scenario_name = scenario.name
        result.corruption_type = scenario.corruption_type
        result.ground_truth = scenario.ground_truth_issues
        result.is_holdout = scenario.is_holdout

        df = scenario.df
        if df is None or df.empty:
            result.error = "No data for scenario"
            return result

        # Prepare data
        df_clean = df.dropna(subset=[target_column])
        if len(df_clean) < 20:
            result.error = "Too few rows after cleaning"
            return result

        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]

        if task_type == "classification":
            y, _ = PipelineBuilder.encode_target(y, "classification")

        # Split
        try:
            X_train, X_test, y_train, y_test = SplitManager.train_test_split(
                X, y, test_size=0.2, random_seed=self.seed, task_type=task_type
            )
        except Exception as e:
            result.error = f"Split failed: {e}"
            return result

        # Quick train for metrics
        from src.experiments.trainer import Trainer
        trainer = Trainer(self.seed)

        try:
            train_result = trainer.train_and_evaluate(
                X_train, X_test, y_train, y_test,
                model_name=model_name, task_type=task_type, n_cv_folds=3
            )
        except Exception as e:
            # Still run audit even if training fails
            train_result = None
            logger.warning(f"Training failed for {scenario.name}: {e}")

        # Run audit
        audit_result = self.audit_engine.run_full_audit(
            df=df,
            target_column=target_column,
            task_type=task_type,
            experiment_id=f"benchmark_{scenario.corruption_type}",
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            train_metrics=train_result.train_metrics if train_result else {},
            test_metrics=train_result.test_metrics if train_result else {},
            cv_metrics=train_result.cv_metrics if train_result else {},
            baseline_metrics=train_result.baseline_metrics if train_result else {},
            feature_importances=train_result.feature_importances if train_result else None,
        )

        result.audit_findings_count = audit_result.total_findings

        # Map findings to categories for comparison
        detected_cats = set()
        for f in audit_result.findings:
            if f.status in ("confirmed", "suspected") and f.severity in ("critical", "high"):
                detected_cats.add(f.category)

        result.detected_categories = list(detected_cats)

        # Compare against ground truth
        gt_set = set(scenario.ground_truth_issues)
        det_set = detected_cats

        result.true_positives = list(gt_set & det_set)
        result.false_negatives = list(gt_set - det_set)
        result.false_positives = list(det_set - gt_set)

        # Calculate per-scenario metrics
        tp = len(result.true_positives)
        fp = len(result.false_positives)
        fn = len(result.false_negatives)

        result.precision = tp / max(tp + fp, 1)
        result.recall = tp / max(tp + fn, 1)
        if result.precision + result.recall > 0:
            result.f1 = 2 * result.precision * result.recall / (result.precision + result.recall)

        return result

    def _calculate_aggregate_metrics(self, aggregate: AggregateBenchmarkResult):
        """Calculate overall precision, recall, F1 across all scenarios."""
        total_tp = 0
        total_fp = 0
        total_fn = 0

        per_category = {}

        for r in aggregate.scenario_results:
            if r.error:
                continue

            total_tp += len(r.true_positives)
            total_fp += len(r.false_positives)
            total_fn += len(r.false_negatives)

            # Track per-category
            for cat in r.true_positives:
                per_category.setdefault(cat, {"tp": 0, "fp": 0, "fn": 0})
                per_category[cat]["tp"] += 1

            for cat in r.false_positives:
                per_category.setdefault(cat, {"tp": 0, "fp": 0, "fn": 0})
                per_category[cat]["fp"] += 1

            for cat in r.false_negatives:
                per_category.setdefault(cat, {"tp": 0, "fp": 0, "fn": 0})
                per_category[cat]["fn"] += 1

        aggregate.overall_precision = total_tp / max(total_tp + total_fp, 1)
        aggregate.overall_recall = total_tp / max(total_tp + total_fn, 1)

        if aggregate.overall_precision + aggregate.overall_recall > 0:
            aggregate.overall_f1 = (
                2 * aggregate.overall_precision * aggregate.overall_recall /
                (aggregate.overall_precision + aggregate.overall_recall)
            )

        total_negative = sum(
            1 for r in aggregate.scenario_results
            if not r.ground_truth and not r.error
        )
        false_positive_scenarios = sum(
            1 for r in aggregate.scenario_results
            if not r.ground_truth and r.false_positives and not r.error
        )
        aggregate.overall_false_positive_rate = (
            false_positive_scenarios / max(total_negative, 1)
        )

        # Per-category F1
        for cat, counts in per_category.items():
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f1 = 2 * p * r / max(p + r, 0.001)
            per_category[cat]["precision"] = round(p, 4)
            per_category[cat]["recall"] = round(r, 4)
            per_category[cat]["f1"] = round(f1, 4)

        aggregate.per_category = per_category

    def _store_results(
        self,
        aggregate: AggregateBenchmarkResult,
        df: pd.DataFrame,
        target_column: str,
    ):
        """Store benchmark results in database."""
        try:
            init_db()
            session = get_session()

            for r in aggregate.scenario_results:
                scenario = BenchmarkRepository.create_scenario(
                    session,
                    base_dataset_id="benchmark_base",
                    scenario_name=r.scenario_name,
                    corruption_type=r.corruption_type,
                    ground_truth_issues=json.dumps(r.ground_truth),
                    random_seed=self.seed,
                    is_holdout=r.is_holdout,
                )

                BenchmarkRepository.create_result(
                    session,
                    scenario_id=scenario.id,
                    detected_issues=json.dumps(r.detected_categories),
                    ground_truth_issues=json.dumps(r.ground_truth),
                    precision_score=r.precision,
                    recall_score=r.recall,
                    f1_score=r.f1,
                    false_positive_rate=0.0,
                    per_category_metrics=json.dumps(aggregate.per_category, default=str),
                )

            session.close()
        except Exception as e:
            logger.error(f"Failed to store benchmark results: {e}")

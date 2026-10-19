"""
Corruption Generator — Creates controlled corrupted datasets for benchmarking.

Generates datasets with known, documented issues to evaluate the auditor's
detection accuracy. All corruptions are applied to REAL datasets.
"""

import logging
from typing import Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from src.config import DEFAULT_RANDOM_SEED

logger = logging.getLogger(__name__)


@dataclass
class CorruptionScenario:
    """A documented corruption scenario."""
    name: str
    corruption_type: str
    description: str
    ground_truth_issues: list[str]  # Expected issue categories
    config: dict = field(default_factory=dict)
    df: Optional[pd.DataFrame] = None
    is_holdout: bool = False


class CorruptionGenerator:
    """
    Generates controlled corrupted datasets from real data.

    Each corruption represents a known evaluation error that the
    auditor should detect. All scenarios are documented with ground truth.
    """

    def __init__(self, random_seed: int = DEFAULT_RANDOM_SEED):
        self.rng = np.random.RandomState(random_seed)
        self.seed = random_seed

    def generate_all_scenarios(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str = "classification",
    ) -> list[CorruptionScenario]:
        """Generate all corruption scenarios from a base dataset."""
        scenarios = []

        # 1. Clean baseline (no corruption)
        scenarios.append(self._clean_baseline(df, target_column))

        # 2. Direct target copy
        scenarios.append(
            self._target_copy(df, target_column)
        )

        # 3. Target-derived feature (encoded)
        scenarios.append(
            self._target_derived_feature(df, target_column, task_type)
        )

        # 4. Duplicate contamination
        scenarios.append(
            self._duplicate_contamination(df, target_column)
        )

        # 5. High-missing column injection
        scenarios.append(
            self._high_missing_injection(df, target_column)
        )

        # 6. Constant column injection
        scenarios.append(
            self._constant_column_injection(df, target_column)
        )

        # 7. Infinite value injection
        scenarios.append(
            self._infinite_value_injection(df, target_column)
        )

        # 8. Near-perfect predictor (holdout scenario)
        scenarios.append(
            self._near_perfect_predictor(df, target_column, task_type)
        )

        return scenarios

    def _clean_baseline(
        self, df: pd.DataFrame, target_column: str
    ) -> CorruptionScenario:
        """Clean baseline — no corruption applied."""
        return CorruptionScenario(
            name="Clean Baseline",
            corruption_type="clean",
            description="Original dataset with no corruption. The auditor should NOT flag critical leakage issues.",
            ground_truth_issues=[],  # Should have no critical findings
            config={"seed": self.seed},
            df=df.copy(),
        )

    def _target_copy(
        self, df: pd.DataFrame, target_column: str
    ) -> CorruptionScenario:
        """Add a column that is an exact copy of the target."""
        df_corrupt = df.copy()
        copy_col_name = f"{target_column}_copy_leaked"
        df_corrupt[copy_col_name] = df_corrupt[target_column].values

        return CorruptionScenario(
            name="Direct Target Copy",
            corruption_type="target_copy",
            description=f"Added column '{copy_col_name}' that is an exact copy of the target.",
            ground_truth_issues=["target_leakage"],
            config={
                "injected_column": copy_col_name,
                "seed": self.seed,
            },
            df=df_corrupt,
        )

    def _target_derived_feature(
        self, df: pd.DataFrame, target_column: str, task_type: str
    ) -> CorruptionScenario:
        """Add a column that is a deterministic transformation of the target."""
        df_corrupt = df.copy()
        derived_col = "derived_feature_leaked"

        target = df_corrupt[target_column]

        if task_type == "classification":
            # Map target values to arbitrary integers
            unique_vals = target.unique()
            mapping = {v: i * 7 + 3 for i, v in enumerate(unique_vals)}
            df_corrupt[derived_col] = target.map(mapping).astype(float)
        else:
            # Linear transform of target
            df_corrupt[derived_col] = target.astype(float) * 2.5 + 10.0

        return CorruptionScenario(
            name="Target-Derived Feature",
            corruption_type="target_derived",
            description=f"Added column '{derived_col}' derived from target via deterministic transform.",
            ground_truth_issues=["target_leakage"],
            config={
                "injected_column": derived_col,
                "transform": "mapping" if task_type == "classification" else "linear",
                "seed": self.seed,
            },
            df=df_corrupt,
        )

    def _duplicate_contamination(
        self, df: pd.DataFrame, target_column: str
    ) -> CorruptionScenario:
        """Inject duplicate rows to create train/test contamination risk."""
        df_corrupt = df.copy()

        # Duplicate 10% of rows
        n_dup = max(1, len(df) // 10)
        dup_indices = self.rng.choice(df.index, size=n_dup, replace=False)
        duplicates = df_corrupt.loc[dup_indices].copy()

        df_corrupt = pd.concat([df_corrupt, duplicates], ignore_index=True)

        return CorruptionScenario(
            name="Duplicate Contamination",
            corruption_type="duplicate_contamination",
            description=f"Injected {n_dup} duplicate rows ({n_dup/len(df)*100:.1f}% of original).",
            ground_truth_issues=["split_contamination", "data_quality"],
            config={
                "n_duplicates": n_dup,
                "original_indices": dup_indices.tolist()[:20],
                "seed": self.seed,
            },
            df=df_corrupt,
        )

    def _high_missing_injection(
        self, df: pd.DataFrame, target_column: str
    ) -> CorruptionScenario:
        """Inject columns with very high missing rates."""
        df_corrupt = df.copy()

        # Add a column with 70% missing
        missing_col = "high_missing_injected"
        values = self.rng.randn(len(df_corrupt))
        mask = self.rng.random(len(df_corrupt)) < 0.7
        values[mask] = np.nan
        df_corrupt[missing_col] = values

        return CorruptionScenario(
            name="High Missing Values",
            corruption_type="high_missing",
            description=f"Added column '{missing_col}' with 70% missing values.",
            ground_truth_issues=["data_quality"],
            config={
                "injected_column": missing_col,
                "missing_pct": 70,
                "seed": self.seed,
            },
            df=df_corrupt,
        )

    def _constant_column_injection(
        self, df: pd.DataFrame, target_column: str
    ) -> CorruptionScenario:
        """Inject constant columns that provide no information."""
        df_corrupt = df.copy()
        df_corrupt["constant_col_injected"] = 42

        return CorruptionScenario(
            name="Constant Column",
            corruption_type="constant_column",
            description="Added a constant column (all values = 42).",
            ground_truth_issues=["data_quality"],
            config={
                "injected_column": "constant_col_injected",
                "constant_value": 42,
                "seed": self.seed,
            },
            df=df_corrupt,
        )

    def _infinite_value_injection(
        self, df: pd.DataFrame, target_column: str
    ) -> CorruptionScenario:
        """Inject infinite values into numeric columns."""
        df_corrupt = df.copy()
        inf_col = "inf_values_injected"
        values = self.rng.randn(len(df_corrupt))
        # Set ~5% to infinity
        inf_mask = self.rng.random(len(df_corrupt)) < 0.05
        values[inf_mask] = np.inf
        df_corrupt[inf_col] = values

        return CorruptionScenario(
            name="Infinite Values",
            corruption_type="infinite_values",
            description=f"Added column '{inf_col}' with ~5% infinite values.",
            ground_truth_issues=["data_quality"],
            config={
                "injected_column": inf_col,
                "inf_pct": 5,
                "seed": self.seed,
            },
            df=df_corrupt,
        )

    def _near_perfect_predictor(
        self, df: pd.DataFrame, target_column: str, task_type: str
    ) -> CorruptionScenario:
        """
        Add a feature that is target + small noise (holdout scenario).
        This is harder to detect than an exact copy.
        """
        df_corrupt = df.copy()
        noisy_col = "noisy_leaky_feature"

        target = df_corrupt[target_column]

        if task_type == "classification":
            # Encode target as numeric then add noise
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            encoded = le.fit_transform(target.astype(str))
            noise = self.rng.normal(0, 0.1, size=len(encoded))
            df_corrupt[noisy_col] = encoded.astype(float) + noise
        else:
            target_numeric = pd.to_numeric(target, errors='coerce')
            std = target_numeric.std()
            noise = self.rng.normal(0, std * 0.05, size=len(target_numeric))
            df_corrupt[noisy_col] = target_numeric + noise

        return CorruptionScenario(
            name="Near-Perfect Predictor (Holdout)",
            corruption_type="noisy_target_copy",
            description=f"Added '{noisy_col}' — target + small noise. Harder to detect.",
            ground_truth_issues=["target_leakage"],
            config={
                "injected_column": noisy_col,
                "noise_level": 0.05,
                "seed": self.seed,
            },
            df=df_corrupt,
            is_holdout=True,
        )

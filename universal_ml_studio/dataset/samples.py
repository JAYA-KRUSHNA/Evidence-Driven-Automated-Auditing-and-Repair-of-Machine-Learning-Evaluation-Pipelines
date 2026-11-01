"""
Sample Dataset Manager.

Provides 7 built-in datasets that work offline (sourced from scikit-learn
and hand-crafted Titanic data).  Each sample is saved as a CSV on first
access, then served from disk on subsequent loads.

Public API
----------
    SampleDatasetManager.list_samples()              → list[dict]
    SampleDatasetManager.load_sample(name)            → (DataFrame, metadata)
    SampleDatasetManager.get_sample_info(name)        → dict
    SampleDatasetManager.ensure_all_samples_exist()   → None
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn import datasets as sk_datasets

from ..config import SAMPLE_DATA_DIR
from .ingestion import DatasetIngestion

logger = logging.getLogger(__name__)


# ── Sample catalogue ──────────────────────────────────────────────────────────
# Each entry has the metadata we expose *and* a loader function.

_SAMPLE_CATALOG: dict[str, dict[str, Any]] = {
    "iris": {
        "display_name": "Iris",
        "task": "classification",
        "description": "3 classes of iris flowers, 4 numerical features.",
        "target_column": "target",
        "num_classes": 3,
    },
    "breast_cancer": {
        "display_name": "Breast Cancer Wisconsin",
        "task": "classification",
        "description": "Binary classification — malignant vs benign tumours, 30 features.",
        "target_column": "target",
        "num_classes": 2,
    },
    "wine": {
        "display_name": "Wine Recognition",
        "task": "classification",
        "description": "3-class classification of Italian wines, 13 features.",
        "target_column": "target",
        "num_classes": 3,
    },
    "diabetes_classification": {
        "display_name": "Diabetes Classification",
        "task": "classification",
        "description": "Binary diabetes prediction from patient measurements.",
        "target_column": "target",
        "num_classes": 2,
    },
    "titanic": {
        "display_name": "Titanic Survival",
        "task": "classification",
        "description": "Binary survival prediction — mixed numerical/categorical features.",
        "target_column": "Survived",
        "num_classes": 2,
    },
    "california_housing": {
        "display_name": "California Housing",
        "task": "regression",
        "description": "Predict median house value from census features.",
        "target_column": "target",
    },
    "diabetes_regression": {
        "display_name": "Diabetes Progression",
        "task": "regression",
        "description": "Predict disease progression one year after baseline.",
        "target_column": "target",
    },
}


# ── Loader functions (one per sample) ─────────────────────────────────────────

def _load_iris() -> pd.DataFrame:
    data = sk_datasets.load_iris(as_frame=True)
    df = data.frame  # type: ignore[attr-defined]
    return df


def _load_breast_cancer() -> pd.DataFrame:
    data = sk_datasets.load_breast_cancer(as_frame=True)
    return data.frame  # type: ignore[attr-defined]


def _load_wine() -> pd.DataFrame:
    data = sk_datasets.load_wine(as_frame=True)
    return data.frame  # type: ignore[attr-defined]


def _load_diabetes_classification() -> pd.DataFrame:
    """
    Derive a binary classification dataset from the sklearn diabetes
    regression dataset by thresholding the target at the median.
    """
    data = sk_datasets.load_diabetes(as_frame=True)
    df = data.frame.copy()  # type: ignore[attr-defined]
    median_target = df["target"].median()
    df["target"] = (df["target"] >= median_target).astype(int)
    return df


def _load_titanic() -> pd.DataFrame:
    """
    Generate a realistic Titanic-style dataset without any internet dependency.
    Uses deterministic random state for reproducibility.
    """
    rng = np.random.RandomState(42)
    n = 891

    pclass = rng.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55])
    sex = rng.choice(["male", "female"], size=n, p=[0.65, 0.35])
    age = rng.normal(loc=30, scale=12, size=n).clip(1, 80).round(1)
    sibsp = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.60, 0.23, 0.10, 0.05, 0.02])
    parch = rng.choice([0, 1, 2, 3], size=n, p=[0.70, 0.15, 0.10, 0.05])
    fare = (rng.exponential(scale=30, size=n) + 5).round(2)
    embarked = rng.choice(["S", "C", "Q"], size=n, p=[0.72, 0.19, 0.09])

    # Survival: higher for female, 1st class, younger
    base_prob = 0.38
    survival_prob = np.full(n, base_prob)
    survival_prob[sex == "female"] += 0.35
    survival_prob[pclass == 1] += 0.15
    survival_prob[age < 16] += 0.10
    survival_prob = survival_prob.clip(0, 1)
    survived = (rng.random(n) < survival_prob).astype(int)

    # Introduce some missing values in Age and Embarked
    age_mask = rng.random(n) < 0.20
    age = age.astype(object)
    age[age_mask] = np.nan

    embarked_mask = rng.random(n) < 0.02
    embarked = embarked.astype(object)
    embarked[embarked_mask] = np.nan

    names = [f"Passenger_{i}" for i in range(1, n + 1)]

    df = pd.DataFrame({
        "PassengerId": range(1, n + 1),
        "Survived": survived,
        "Pclass": pclass,
        "Name": names,
        "Sex": sex,
        "Age": pd.to_numeric(age, errors="coerce"),
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked,
    })
    return df


def _load_california_housing() -> pd.DataFrame:
    data = sk_datasets.fetch_california_housing(as_frame=True)
    df = data.frame.copy()  # type: ignore[attr-defined]
    # Rename the target column to "target" for consistency
    if "MedHouseVal" in df.columns:
        df = df.rename(columns={"MedHouseVal": "target"})
    return df


def _load_diabetes_regression() -> pd.DataFrame:
    data = sk_datasets.load_diabetes(as_frame=True)
    return data.frame  # type: ignore[attr-defined]


_LOADERS: dict[str, callable] = {
    "iris": _load_iris,
    "breast_cancer": _load_breast_cancer,
    "wine": _load_wine,
    "diabetes_classification": _load_diabetes_classification,
    "titanic": _load_titanic,
    "california_housing": _load_california_housing,
    "diabetes_regression": _load_diabetes_regression,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Public class
# ═══════════════════════════════════════════════════════════════════════════════

class SampleDatasetManager:
    """Manage built-in sample datasets — generate once, load from disk."""

    @staticmethod
    def list_samples() -> list[dict[str, Any]]:
        """Return metadata for all available sample datasets."""
        result = []
        for key, info in _SAMPLE_CATALOG.items():
            csv_path = SAMPLE_DATA_DIR / f"{key}.csv"
            entry = {
                "name": key,
                "available_locally": csv_path.exists(),
                **info,
            }
            result.append(entry)
        return result

    @staticmethod
    def get_sample_info(name: str) -> dict[str, Any]:
        """Return metadata for a single sample dataset."""
        name = name.lower().strip()
        if name not in _SAMPLE_CATALOG:
            raise ValueError(
                f"Unknown sample dataset '{name}'. "
                f"Available: {list(_SAMPLE_CATALOG.keys())}"
            )
        csv_path = SAMPLE_DATA_DIR / f"{name}.csv"
        return {"name": name, "available_locally": csv_path.exists(), **_SAMPLE_CATALOG[name]}

    @staticmethod
    def load_sample(name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Load a sample dataset.  If the CSV doesn't exist yet, generate it
        from scikit-learn and cache it to disk.

        Returns (DataFrame, metadata).
        """
        name = name.lower().strip()
        if name not in _LOADERS:
            raise ValueError(
                f"Unknown sample dataset '{name}'. "
                f"Available: {list(_LOADERS.keys())}"
            )

        csv_path = SAMPLE_DATA_DIR / f"{name}.csv"

        if not csv_path.exists():
            logger.info("Generating sample dataset '%s' → %s", name, csv_path)
            df = _LOADERS[name]()
            df.to_csv(csv_path, index=False)
        else:
            logger.info("Loading cached sample dataset '%s'", name)

        # Always read through DatasetIngestion for consistent metadata
        df, metadata = DatasetIngestion.read(csv_path)
        metadata["source"] = "sample"
        metadata["sample_name"] = name
        metadata["task"] = _SAMPLE_CATALOG[name]["task"]
        metadata["target_column"] = _SAMPLE_CATALOG[name]["target_column"]
        return df, metadata

    @staticmethod
    def ensure_all_samples_exist() -> None:
        """Generate all sample datasets that don't yet exist on disk."""
        for name in _LOADERS:
            csv_path = SAMPLE_DATA_DIR / f"{name}.csv"
            if not csv_path.exists():
                logger.info("Pre-generating sample: %s", name)
                df = _LOADERS[name]()
                df.to_csv(csv_path, index=False)
        logger.info("All %d sample datasets are available.", len(_LOADERS))

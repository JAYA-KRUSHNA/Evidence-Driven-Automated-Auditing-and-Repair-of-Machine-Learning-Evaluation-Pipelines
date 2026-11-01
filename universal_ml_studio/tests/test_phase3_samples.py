"""
Phase 3 — Tests for dataset/samples.py

Covers:
    - List all samples
    - Load each sample dataset
    - Verify metadata correctness
    - Error on unknown sample name
    - Offline caching (file exists on disk after first load)
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from universal_ml_studio.dataset.samples import SampleDatasetManager


class TestSampleList:

    def test_list_returns_7_samples(self):
        samples = SampleDatasetManager.list_samples()
        assert len(samples) == 7

    def test_list_has_required_keys(self):
        for s in SampleDatasetManager.list_samples():
            assert "name" in s
            assert "display_name" in s
            assert "task" in s
            assert "target_column" in s


class TestSampleInfo:

    def test_known_sample(self):
        info = SampleDatasetManager.get_sample_info("iris")
        assert info["task"] == "classification"
        assert info["target_column"] == "target"

    def test_unknown_sample(self):
        with pytest.raises(ValueError, match="Unknown sample"):
            SampleDatasetManager.get_sample_info("nonexistent")


class TestSampleLoad:

    @pytest.mark.parametrize("name,expected_task", [
        ("iris", "classification"),
        ("breast_cancer", "classification"),
        ("wine", "classification"),
        ("diabetes_classification", "classification"),
        ("titanic", "classification"),
        ("california_housing", "regression"),
        ("diabetes_regression", "regression"),
    ])
    def test_load_each_sample(self, name, expected_task):
        df, meta = SampleDatasetManager.load_sample(name)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert meta["task"] == expected_task
        assert meta["source"] == "sample"
        assert meta["num_rows"] == len(df)
        assert meta["target_column"] in df.columns

    def test_unknown_sample_load(self):
        with pytest.raises(ValueError, match="Unknown sample"):
            SampleDatasetManager.load_sample("fake_dataset")

    def test_cached_on_disk(self):
        """After loading, the CSV should exist in sample_datasets/."""
        from universal_ml_studio.config import SAMPLE_DATA_DIR
        SampleDatasetManager.load_sample("iris")
        assert (SAMPLE_DATA_DIR / "iris.csv").exists()


class TestEnsureAll:

    def test_ensure_all_samples_exist(self):
        from universal_ml_studio.config import SAMPLE_DATA_DIR
        SampleDatasetManager.ensure_all_samples_exist()
        for name in ["iris", "breast_cancer", "wine", "diabetes_classification",
                      "titanic", "california_housing", "diabetes_regression"]:
            assert (SAMPLE_DATA_DIR / f"{name}.csv").exists(), f"{name}.csv not found"

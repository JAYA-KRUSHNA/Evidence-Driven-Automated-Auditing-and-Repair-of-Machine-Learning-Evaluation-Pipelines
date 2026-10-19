"""
Test fixtures shared across all tests.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def clean_classification_df():
    """A clean binary classification dataset with no leakage."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n),
        "feature_3": np.random.uniform(0, 10, n),
        "category": np.random.choice(["A", "B", "C"], n),
        "target": np.random.choice([0, 1], n),
    })
    return df


@pytest.fixture
def leaky_classification_df():
    """A dataset with a direct target-copy leakage column."""
    np.random.seed(42)
    n = 200
    target = np.random.choice([0, 1], n)
    df = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n),
        "leaked_copy": target,  # Direct copy of target
        "target": target,
    })
    return df


@pytest.fixture
def duplicate_df():
    """A dataset with duplicate rows."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n),
        "target": np.random.choice([0, 1], n),
    })
    # Duplicate 20% of rows
    dups = df.sample(20, random_state=42)
    df = pd.concat([df, dups], ignore_index=True)
    return df


@pytest.fixture
def regression_df():
    """A clean regression dataset."""
    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    target = 2 * x1 + 3 * x2 + np.random.randn(n) * 0.5
    df = pd.DataFrame({
        "feature_1": x1,
        "feature_2": x2,
        "feature_3": np.random.randn(n),
        "target": target,
    })
    return df


@pytest.fixture
def missing_df():
    """A dataset with various missing value patterns."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n),
        "high_missing": np.where(np.random.random(n) < 0.7, np.nan, np.random.randn(n)),
        "some_missing": np.where(np.random.random(n) < 0.1, np.nan, np.random.randn(n)),
        "constant_col": 42,
        "target": np.random.choice([0, 1], n),
    })
    return df

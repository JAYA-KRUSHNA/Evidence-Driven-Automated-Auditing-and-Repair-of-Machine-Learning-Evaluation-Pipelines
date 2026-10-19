"""
AI ML Experiment Auditor - Configuration

Central configuration for paths, constants, and application settings.
All paths are relative to the project root.
"""

import os
from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DB_DIR = PROJECT_ROOT / "data"

# Ensure directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Database ───────────────────────────────────────────────────────────────────

DATABASE_URL = f"sqlite:///{DB_DIR / 'auditor.db'}"

# ── Dataset Defaults ───────────────────────────────────────────────────────────

MAX_UPLOAD_SIZE_MB = 200
SUPPORTED_UPLOAD_EXTENSIONS = {".csv", ".parquet"}
MAX_PREVIEW_ROWS = 100

# ── Experiment Defaults ────────────────────────────────────────────────────────

DEFAULT_RANDOM_SEED = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_FOLDS = 5
MAX_TRAINING_ROWS = 100_000  # Sample if dataset exceeds this

SUPPORTED_CLASSIFIERS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest Classifier",
    "gradient_boosting": "Gradient Boosting Classifier",
    "decision_tree": "Decision Tree Classifier",
    "svm": "Linear SVM Classifier",
    "knn": "K-Nearest Neighbors Classifier",
}

SUPPORTED_REGRESSORS = {
    "linear_regression": "Linear Regression",
    "random_forest_reg": "Random Forest Regressor",
    "gradient_boosting_reg": "Gradient Boosting Regressor",
    "decision_tree_reg": "Decision Tree Regressor",
    "svr": "Linear SVR",
    "knn_reg": "K-Nearest Neighbors Regressor",
}

# ── Audit Thresholds ──────────────────────────────────────────────────────────

# Target leakage
TARGET_LEAKAGE_CORRELATION_THRESHOLD = 0.98
TARGET_LEAKAGE_MUTUAL_INFO_THRESHOLD = 0.95

# Split contamination
OVERLAP_WARNING_THRESHOLD = 0.01  # >1% overlap is suspicious
NEAR_DUPLICATE_JACCARD_THRESHOLD = 0.95

# Performance anomaly
TRAIN_TEST_GAP_THRESHOLD = 0.15  # >15% gap is suspicious
SUSPICIOUSLY_HIGH_ACCURACY = 0.99
CV_VARIANCE_THRESHOLD = 0.05  # High variance across folds

# Data quality
MISSING_VALUE_WARNING_THRESHOLD = 0.05  # >5% missing
MISSING_VALUE_CRITICAL_THRESHOLD = 0.50  # >50% missing
CLASS_IMBALANCE_THRESHOLD = 0.1  # Minority class < 10%

# ── Severity Levels ───────────────────────────────────────────────────────────

class Severity:
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

# ── Finding Statuses ──────────────────────────────────────────────────────────

class FindingStatus:
    CONFIRMED = "confirmed"
    SUSPECTED = "suspected"
    INFORMATIONAL = "informational"
    INCONCLUSIVE = "inconclusive"

# ── Audit Categories ──────────────────────────────────────────────────────────

class AuditCategory:
    TARGET_LEAKAGE = "target_leakage"
    PREPROCESSING_LEAKAGE = "preprocessing_leakage"
    SPLIT_CONTAMINATION = "split_contamination"
    SPLIT_STRATEGY = "split_strategy"
    PERFORMANCE_ANOMALY = "performance_anomaly"
    EVALUATION_INTEGRITY = "evaluation_integrity"
    DATA_QUALITY = "data_quality"

# ── Software Version Tracking ─────────────────────────────────────────────────

def get_software_versions() -> dict:
    """Capture current software versions for reproducibility."""
    import sklearn
    import pandas as pd
    import numpy as np
    import platform
    import sys

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }

"""
Global configuration for Universal AI/ML Studio.

Centralises every path, constant, and default value so that no other module
needs to hardcode anything.  All storage directories are created on import.
"""

from pathlib import Path

# ── Project Root ───────────────────────────────────────────────────────────────
# The directory that contains *this* file (universal_ml_studio/).
PROJECT_ROOT = Path(__file__).parent.resolve()

# ── Storage Directories ───────────────────────────────────────────────────────
STORAGE_DIR       = PROJECT_ROOT / "storage"
DATASETS_DIR      = STORAGE_DIR / "datasets"
MODELS_DIR        = STORAGE_DIR / "models"
REPORTS_DIR       = STORAGE_DIR / "reports"
SAMPLE_DATA_DIR   = STORAGE_DIR / "sample_datasets"

# Create all storage directories on first import
for _d in (STORAGE_DIR, DATASETS_DIR, MODELS_DIR, REPORTS_DIR, SAMPLE_DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Database ───────────────────────────────────────────────────────────────────
DATABASE_PATH = STORAGE_DIR / "studio.db"
DATABASE_URL  = f"sqlite:///{DATABASE_PATH}"

# ── Upload Limits ──────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = 200          # reject files larger than this
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

# ── Validation Thresholds ──────────────────────────────────────────────────────
MIN_ROWS_FOR_ML            = 30          # fewer rows → INVALID
MIN_FEATURE_COLUMNS        = 1           # at least one feature besides target
HIGH_CARDINALITY_THRESHOLD = 50          # unique-value count to flag a column
CONSTANT_COLUMN_NUNIQUE    = 1           # columns with only 1 unique value
MISSING_VALUE_WARNING_PCT  = 5.0         # >5 % missing → warning
MISSING_VALUE_ERROR_PCT    = 80.0        # >80 % missing → error
CLASS_IMBALANCE_RATIO      = 10.0        # majority/minority ratio to warn
MIN_SAMPLES_PER_CLASS      = 5           # fewer → warning
OUTLIER_IQR_MULTIPLIER     = 1.5
OUTLIER_ZSCORE_THRESHOLD   = 3.0
CORRELATION_LEAKAGE_THRESHOLD = 0.98     # suspiciously high correlation

# ── ML Defaults ────────────────────────────────────────────────────────────────
DEFAULT_TEST_SIZE   = 0.2
DEFAULT_CV_FOLDS    = 5
DEFAULT_RANDOM_SEED = 42
MAX_UNIQUE_FOR_CLASSIFICATION = 25       # heuristic: ≤25 unique target values → classification

# ── Supported Algorithms ───────────────────────────────────────────────────────
CLASSIFICATION_ALGORITHMS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "knn",
    "svm",
    "naive_bayes",
    "gradient_boosting",
    "extra_trees",
    "sgd",
]

REGRESSION_ALGORITHMS = [
    "linear_regression",
    "ridge",
    "lasso",
    "decision_tree",
    "random_forest",
    "knn",
    "svr",
    "gradient_boosting",
    "extra_trees",
    "sgd",
]

# ── Report Formats ─────────────────────────────────────────────────────────────
REPORT_FORMATS = ["pdf", "csv", "json"]

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_DATE   = "%Y-%m-%d %H:%M:%S"

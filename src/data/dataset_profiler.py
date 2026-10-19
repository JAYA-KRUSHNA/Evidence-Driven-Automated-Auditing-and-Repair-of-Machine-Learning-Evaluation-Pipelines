"""
Dataset Profiler — Statistical profiling of datasets.

Generates a comprehensive profile of a DataFrame including:
- Column types and distributions
- Missing value analysis
- Unique value counts
- Statistical summaries
- Target column analysis
- Data quality indicators
"""

import logging
from typing import Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ColumnProfile:
    """Profile for a single column."""
    name: str
    dtype: str
    pandas_dtype: str
    semantic_type: str  # "numeric", "categorical", "datetime", "text", "binary", "identifier"
    count: int = 0
    missing_count: int = 0
    missing_pct: float = 0.0
    unique_count: int = 0
    unique_pct: float = 0.0
    # Numeric stats
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    median: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    skewness: Optional[float] = None
    has_infinity: bool = False
    has_negative: bool = False
    # Categorical stats
    top_values: Optional[dict] = None
    # Quality indicators
    is_constant: bool = False
    is_potential_id: bool = False
    is_potential_target: bool = False

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class DatasetProfile:
    """Complete profile for a dataset."""
    row_count: int = 0
    col_count: int = 0
    memory_mb: float = 0.0
    missing_cells_total: int = 0
    missing_cells_pct: float = 0.0
    duplicate_rows: int = 0
    duplicate_pct: float = 0.0
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)
    potential_id_columns: list[str] = field(default_factory=list)
    target_candidates: list[str] = field(default_factory=list)
    column_profiles: dict[str, ColumnProfile] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        result = {k: v for k, v in self.__dict__.items() if k != "column_profiles"}
        result["column_profiles"] = {
            name: cp.to_dict() for name, cp in self.column_profiles.items()
        }
        return result


class DatasetProfiler:
    """
    Generates comprehensive statistical profiles for DataFrames.
    All analysis is based on actual data — no fabrication.
    """

    def profile(self, df: pd.DataFrame,
                target_column: Optional[str] = None) -> DatasetProfile:
        """
        Generate a complete profile of a DataFrame.

        Args:
            df: The DataFrame to profile
            target_column: Optional known target column

        Returns:
            DatasetProfile with all statistics
        """
        profile = DatasetProfile()
        profile.row_count = len(df)
        profile.col_count = len(df.columns)
        profile.memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

        # Overall missing
        total_cells = profile.row_count * profile.col_count
        profile.missing_cells_total = int(df.isna().sum().sum())
        profile.missing_cells_pct = round(
            profile.missing_cells_total / max(total_cells, 1) * 100, 2
        )

        # Duplicates
        profile.duplicate_rows = int(df.duplicated().sum())
        profile.duplicate_pct = round(
            profile.duplicate_rows / max(profile.row_count, 1) * 100, 2
        )

        # Profile each column
        for col in df.columns:
            cp = self._profile_column(df, col)
            profile.column_profiles[col] = cp

            # Categorize
            if cp.semantic_type == "numeric":
                profile.numeric_columns.append(col)
            elif cp.semantic_type in ("categorical", "binary"):
                profile.categorical_columns.append(col)
            elif cp.semantic_type == "datetime":
                profile.datetime_columns.append(col)

            if cp.is_constant:
                profile.constant_columns.append(col)
            if cp.is_potential_id:
                profile.potential_id_columns.append(col)

        # Identify target candidates
        profile.target_candidates = self._identify_target_candidates(
            df, profile, target_column
        )

        # Generate warnings
        profile.warnings = self._generate_warnings(profile)

        return profile

    def _profile_column(self, df: pd.DataFrame, col: str) -> ColumnProfile:
        """Profile a single column."""
        series = df[col]
        non_null = series.dropna()

        cp = ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            pandas_dtype=series.dtype.name,
            semantic_type=self._detect_semantic_type(series),
            count=len(non_null),
            missing_count=int(series.isna().sum()),
            missing_pct=round(series.isna().mean() * 100, 2),
            unique_count=int(series.nunique()),
            unique_pct=round(series.nunique() / max(len(non_null), 1) * 100, 2),
        )

        # Check if constant
        cp.is_constant = cp.unique_count <= 1

        # Check if potential identifier
        cp.is_potential_id = (
            cp.unique_pct > 95 and
            len(non_null) > 10 and
            not pd.api.types.is_float_dtype(series)
        )

        # Numeric stats
        if cp.semantic_type == "numeric" and len(non_null) > 0:
            try:
                numeric_vals = pd.to_numeric(non_null, errors='coerce').dropna()
                if len(numeric_vals) > 0:
                    cp.mean = round(float(numeric_vals.mean()), 4)
                    cp.std = round(float(numeric_vals.std()), 4)
                    cp.min_val = float(numeric_vals.min())
                    cp.max_val = float(numeric_vals.max())
                    cp.median = float(numeric_vals.median())
                    cp.q25 = float(numeric_vals.quantile(0.25))
                    cp.q75 = float(numeric_vals.quantile(0.75))
                    cp.skewness = round(float(numeric_vals.skew()), 4)
                    cp.has_infinity = bool(np.isinf(numeric_vals).any())
                    cp.has_negative = bool((numeric_vals < 0).any())
            except (ValueError, TypeError):
                pass

        # Categorical stats (top values)
        if cp.semantic_type in ("categorical", "binary") and len(non_null) > 0:
            top = non_null.value_counts().head(10)
            cp.top_values = {str(k): int(v) for k, v in top.items()}

        return cp

    def _detect_semantic_type(self, series: pd.Series) -> str:
        """Detect the semantic type of a column."""
        dtype = series.dtype

        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "datetime"

        # Check for numeric
        if pd.api.types.is_numeric_dtype(dtype):
            non_null = series.dropna()
            unique = non_null.nunique()

            # Binary numeric
            if unique == 2:
                return "binary"

            # Few unique integers might be categorical
            if pd.api.types.is_integer_dtype(dtype) and unique <= 10 and len(non_null) > 20:
                return "categorical"

            return "numeric"

        # Check for boolean
        if pd.api.types.is_bool_dtype(dtype):
            return "binary"

        # String/object
        non_null = series.dropna()
        if len(non_null) > 0:
            unique = non_null.nunique()

            # Binary
            if unique == 2:
                return "binary"

            # Potential datetime strings
            try:
                sample = non_null.head(20)
                pd.to_datetime(sample, format="mixed")
                return "datetime"
            except (ValueError, TypeError):
                pass

            # Check if all unique (identifier)
            if unique == len(non_null) and len(non_null) > 10:
                return "identifier"

            # Categorical if reasonable number of unique values
            if unique <= 50 or unique / max(len(non_null), 1) < 0.05:
                return "categorical"

            return "text"

        return "categorical"

    def _identify_target_candidates(self, df: pd.DataFrame,
                                     profile: DatasetProfile,
                                     known_target: Optional[str]) -> list[str]:
        """Identify columns that could serve as prediction targets."""
        candidates = []

        if known_target and known_target in df.columns:
            candidates.append(known_target)

        for col, cp in profile.column_profiles.items():
            if col in candidates:
                continue
            # Skip identifiers and constants
            if cp.is_potential_id or cp.is_constant:
                continue
            # Skip high-missing columns
            if cp.missing_pct > 50:
                continue
            # Good targets: binary, categorical with few classes, or numeric
            if cp.semantic_type == "binary":
                cp.is_potential_target = True
                candidates.append(col)
            elif cp.semantic_type == "categorical" and cp.unique_count <= 20:
                cp.is_potential_target = True
                candidates.append(col)
            elif cp.semantic_type == "numeric" and not cp.is_potential_id:
                cp.is_potential_target = True
                candidates.append(col)

        return candidates

    def _generate_warnings(self, profile: DatasetProfile) -> list[str]:
        """Generate data quality warnings based on the profile."""
        warnings = []

        if profile.row_count < 50:
            warnings.append(
                f"Very small dataset ({profile.row_count} rows). "
                "Results may be unreliable."
            )

        if profile.missing_cells_pct > 20:
            warnings.append(
                f"High overall missing value rate: {profile.missing_cells_pct}%"
            )

        if profile.duplicate_pct > 5:
            warnings.append(
                f"Notable duplicate rows: {profile.duplicate_rows} "
                f"({profile.duplicate_pct}%)"
            )

        if profile.constant_columns:
            warnings.append(
                f"Constant columns (no information): "
                f"{', '.join(profile.constant_columns)}"
            )

        # Check for columns with very high missing rates
        high_missing = [
            f"{name} ({cp.missing_pct}%)"
            for name, cp in profile.column_profiles.items()
            if cp.missing_pct > 50
        ]
        if high_missing:
            warnings.append(
                f"Columns with >50% missing: {', '.join(high_missing)}"
            )

        # Check for infinity
        inf_cols = [
            name for name, cp in profile.column_profiles.items()
            if cp.has_infinity
        ]
        if inf_cols:
            warnings.append(
                f"Columns containing infinity values: {', '.join(inf_cols)}"
            )

        return warnings

    def get_target_analysis(self, df: pd.DataFrame,
                            target_column: str) -> dict:
        """Analyze a specific target column in detail."""
        if target_column not in df.columns:
            return {"error": f"Column '{target_column}' not found."}

        series = df[target_column].dropna()
        analysis = {
            "column": target_column,
            "count": len(series),
            "missing": int(df[target_column].isna().sum()),
            "unique": int(series.nunique()),
        }

        if pd.api.types.is_numeric_dtype(series):
            if series.nunique() <= 20:
                # Classification target
                analysis["task_type"] = "classification"
                vc = series.value_counts()
                analysis["class_distribution"] = {
                    str(k): int(v) for k, v in vc.items()
                }
                analysis["class_balance"] = {
                    str(k): round(v / len(series) * 100, 2)
                    for k, v in vc.items()
                }
                minority_pct = min(analysis["class_balance"].values())
                analysis["is_imbalanced"] = minority_pct < 10
            else:
                # Regression target
                analysis["task_type"] = "regression"
                analysis["mean"] = round(float(series.mean()), 4)
                analysis["std"] = round(float(series.std()), 4)
                analysis["min"] = float(series.min())
                analysis["max"] = float(series.max())
                analysis["median"] = float(series.median())
        else:
            # Categorical target → classification
            analysis["task_type"] = "classification"
            vc = series.value_counts()
            analysis["class_distribution"] = {
                str(k): int(v) for k, v in vc.items()
            }
            analysis["class_balance"] = {
                str(k): round(v / len(series) * 100, 2)
                for k, v in vc.items()
            }
            minority_pct = min(analysis["class_balance"].values()) if analysis["class_balance"] else 0
            analysis["is_imbalanced"] = minority_pct < 10

        return analysis

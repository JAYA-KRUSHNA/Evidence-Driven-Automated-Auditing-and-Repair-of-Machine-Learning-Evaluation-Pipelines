"""
Dataset Ingestion Module.

Reads CSV / Excel (.xlsx, .xls) files and extracts rich metadata
without hard-coding any column names or dataset structure.

Public API
----------
    DatasetIngestion.read(file_path)       → (DataFrame, metadata_dict)
    DatasetIngestion.extract_metadata(df)  → metadata_dict
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import SUPPORTED_EXTENSIONS, HIGH_CARDINALITY_THRESHOLD

logger = logging.getLogger(__name__)


class DatasetReadError(Exception):
    """Raised when a dataset file cannot be read."""


class DatasetIngestion:
    """Stateless helper for reading datasets and extracting metadata."""

    # ── Read ──────────────────────────────────────────────────────────────

    @staticmethod
    def read(file_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Read a dataset file and return (DataFrame, metadata).

        Raises DatasetReadError on any I/O or format problem.
        """
        path = Path(file_path)

        # existence
        if not path.exists():
            raise DatasetReadError(f"File not found: {path}")

        # extension
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise DatasetReadError(
                f"Unsupported format '{ext}'. Supported: {SUPPORTED_EXTENSIONS}"
            )

        # size check (0 bytes)
        if path.stat().st_size == 0:
            raise DatasetReadError(f"File is empty: {path}")

        try:
            if ext == ".csv":
                df = pd.read_csv(path, low_memory=False)
            elif ext == ".xlsx":
                df = pd.read_excel(path, engine="openpyxl")
            elif ext == ".xls":
                df = pd.read_excel(path, engine="xlrd")
            else:
                raise DatasetReadError(f"Unsupported extension: {ext}")
        except DatasetReadError:
            raise
        except Exception as exc:
            raise DatasetReadError(f"Failed to read {path.name}: {exc}") from exc

        if df.empty:
            raise DatasetReadError(f"Dataset is empty (0 rows): {path.name}")

        metadata = DatasetIngestion.extract_metadata(df, file_path=path)
        logger.info(
            "Read %s — %d rows × %d cols", path.name, df.shape[0], df.shape[1]
        )
        return df, metadata

    # ── Metadata extraction ───────────────────────────────────────────────

    @staticmethod
    def extract_metadata(
        df: pd.DataFrame,
        file_path: Path | None = None,
    ) -> dict[str, Any]:
        """
        Analyse a DataFrame and return a rich metadata dictionary.
        Nothing is hard-coded — works on any tabular dataset.
        """
        num_rows, num_cols = df.shape
        column_names = df.columns.tolist()

        # dtype mapping  (safe string representation)
        column_types: dict[str, str] = {
            col: str(df[col].dtype) for col in column_names
        }

        # Numerical vs categorical
        # Include "str" for Pandas 3.x which uses StringDtype by default
        numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "string", "str", "category", "bool"]
        ).columns.tolist()

        # Datetime columns
        datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
        # Also try to detect date-like string/object columns
        inferred_datetime_cols = DatasetIngestion._detect_datetime_columns(df)
        datetime_cols = list(set(datetime_cols + inferred_datetime_cols))

        # Identifier columns (unique, non-numeric, monotonic, or named id/key)
        identifier_cols = DatasetIngestion._detect_identifier_columns(df)

        # Missing values
        missing_counts = df.isnull().sum()
        missing_info = {
            col: {"count": int(missing_counts[col]),
                  "percentage": round(float(missing_counts[col] / num_rows * 100), 2)}
            for col in column_names if missing_counts[col] > 0
        }

        # Memory usage
        memory_bytes = int(df.memory_usage(deep=True).sum())

        meta: dict[str, Any] = {
            "num_rows": num_rows,
            "num_columns": num_cols,
            "column_names": column_names,
            "column_types": column_types,
            "numerical_columns": numerical_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "identifier_columns": identifier_cols,
            "missing_values": missing_info,
            "total_missing_cells": int(missing_counts.sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_bytes": memory_bytes,
        }

        if file_path is not None:
            p = Path(file_path)
            meta["file_name"] = p.name
            meta["file_format"] = p.suffix.lstrip(".").lower()
            meta["file_size_bytes"] = p.stat().st_size

        return meta

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _detect_datetime_columns(df: pd.DataFrame) -> list[str]:
        """
        Heuristically detect object columns that are likely dates.
        Tries pd.to_datetime on a small sample; if ≥80 % parse → date column.
        """
        candidates: list[str] = []
        for col in df.select_dtypes(include=["object", "string", "str"]).columns:
            sample = df[col].dropna().head(50)
            if sample.empty:
                continue
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                success_rate = parsed.notna().mean()
                if success_rate >= 0.8:
                    candidates.append(col)
            except Exception:
                continue
        return candidates

    @staticmethod
    def _detect_identifier_columns(df: pd.DataFrame) -> list[str]:
        """
        Detect columns that look like row identifiers:
        - name contains 'id', 'key', 'index', 'code' (case-insensitive)
        - all values unique
        - monotonically increasing integers
        """
        ids: list[str] = []
        id_patterns = {"id", "key", "index", "code", "identifier", "uuid"}

        for col in df.columns:
            col_lower = col.lower().strip()

            # name-based heuristic
            is_name_match = any(pat in col_lower for pat in id_patterns)

            # uniqueness check
            nunique = df[col].nunique(dropna=True)
            is_unique = nunique == len(df)

            # monotonic integer check
            is_monotonic = False
            if df[col].dtype in (np.int64, np.int32, np.float64):
                try:
                    is_monotonic = df[col].is_monotonic_increasing or df[col].is_monotonic_decreasing
                except Exception:
                    pass

            if (is_name_match and is_unique) or (is_unique and is_monotonic):
                ids.append(col)

        return ids

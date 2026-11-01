"""
Dataset Registry.

Persistent CRUD layer for datasets — upload, retrieve, version,
and delete datasets through the SQLite database.

Public API
----------
    DatasetRegistry.register(df, metadata, ...)     → Dataset row
    DatasetRegistry.get(dataset_id)                 → Dataset row
    DatasetRegistry.list_all()                      → list[Dataset]
    DatasetRegistry.load_dataframe(dataset_id)      → DataFrame
    DatasetRegistry.create_version(dataset_id, df)  → DatasetVersion
    DatasetRegistry.delete(dataset_id)              → None
    DatasetRegistry.update_validation_status(...)   → None
"""

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..config import DATASETS_DIR
from ..database.connection import get_session
from ..database.models import Dataset, DatasetVersion
from .ingestion import DatasetIngestion

logger = logging.getLogger(__name__)


class DatasetNotFoundError(Exception):
    """Raised when a dataset ID cannot be found in the registry."""


class DatasetRegistry:
    """CRUD operations for dataset metadata and files."""

    # ── Register a new dataset ────────────────────────────────────────────

    @staticmethod
    def register(
        df: pd.DataFrame,
        metadata: dict[str, Any],
        name: str | None = None,
        source: str = "upload",
        original_file_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Save a DataFrame to the datasets directory, record it in the DB,
        and return the dataset record as a dict.
        """
        session = get_session()
        try:
            original_name = metadata.get("file_name", "dataset.csv")
            dataset_name = name or Path(original_name).stem

            # Create a unique directory for this dataset
            ds = Dataset(
                name=dataset_name,
                original_name=original_name,
                file_path="",           # will update after saving
                file_format=metadata.get("file_format", "csv"),
                source=source,
                num_rows=metadata.get("num_rows", len(df)),
                num_columns=metadata.get("num_columns", len(df.columns)),
                column_names=metadata.get("column_names", df.columns.tolist()),
                column_types=metadata.get("column_types", {}),
                numerical_cols=metadata.get("numerical_columns", []),
                categorical_cols=metadata.get("categorical_columns", []),
                datetime_cols=metadata.get("datetime_columns", []),
                identifier_cols=metadata.get("identifier_columns", []),
                file_size_bytes=metadata.get("file_size_bytes", 0),
                validation_status="pending",
            )
            session.add(ds)
            session.flush()  # get the auto-generated id

            # Save the file
            ds_dir = DATASETS_DIR / ds.id
            ds_dir.mkdir(parents=True, exist_ok=True)
            csv_path = ds_dir / f"{dataset_name}.csv"
            df.to_csv(csv_path, index=False)
            ds.file_path = str(csv_path)
            ds.file_size_bytes = csv_path.stat().st_size

            # Also copy original if provided
            if original_file_path and Path(original_file_path).exists():
                orig_dest = ds_dir / f"original_{Path(original_file_path).name}"
                shutil.copy2(str(original_file_path), str(orig_dest))

            # Create version 1
            v1 = DatasetVersion(
                dataset_id=ds.id,
                version=1,
                file_path=str(csv_path),
                num_rows=len(df),
                num_columns=len(df.columns),
                description="Initial upload",
            )
            session.add(v1)
            session.commit()

            result = DatasetRegistry._to_dict(ds)
            logger.info("Registered dataset '%s' (id=%s)", dataset_name, ds.id)
            return result

        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ── Get by ID ─────────────────────────────────────────────────────────

    @staticmethod
    def get(dataset_id: str) -> dict[str, Any]:
        """Retrieve dataset metadata by ID."""
        session = get_session()
        try:
            ds = session.query(Dataset).filter_by(id=dataset_id).first()
            if ds is None:
                raise DatasetNotFoundError(f"Dataset '{dataset_id}' not found")
            return DatasetRegistry._to_dict(ds)
        finally:
            session.close()

    # ── List all ──────────────────────────────────────────────────────────

    @staticmethod
    def list_all() -> list[dict[str, Any]]:
        """Return all registered datasets."""
        session = get_session()
        try:
            datasets = session.query(Dataset).order_by(Dataset.upload_time.desc()).all()
            return [DatasetRegistry._to_dict(ds) for ds in datasets]
        finally:
            session.close()

    # ── Load DataFrame ────────────────────────────────────────────────────

    @staticmethod
    def load_dataframe(dataset_id: str) -> pd.DataFrame:
        """Load the dataset's CSV into a DataFrame."""
        info = DatasetRegistry.get(dataset_id)
        path = Path(info["file_path"])
        if not path.exists():
            raise DatasetNotFoundError(f"Dataset file not found: {path}")
        df, _ = DatasetIngestion.read(path)
        return df

    # ── Create a new version ──────────────────────────────────────────────

    @staticmethod
    def create_version(
        dataset_id: str,
        df: pd.DataFrame,
        description: str = "",
    ) -> dict[str, Any]:
        """
        Save a modified DataFrame as a new version of an existing dataset.
        Does NOT overwrite the original.
        """
        session = get_session()
        try:
            ds = session.query(Dataset).filter_by(id=dataset_id).first()
            if ds is None:
                raise DatasetNotFoundError(f"Dataset '{dataset_id}' not found")

            # determine next version number
            max_ver = (
                session.query(DatasetVersion)
                .filter_by(dataset_id=dataset_id)
                .count()
            )
            next_ver = max_ver + 1

            ds_dir = DATASETS_DIR / dataset_id
            ds_dir.mkdir(parents=True, exist_ok=True)
            version_path = ds_dir / f"{ds.name}_v{next_ver}.csv"
            df.to_csv(version_path, index=False)

            ver = DatasetVersion(
                dataset_id=dataset_id,
                version=next_ver,
                file_path=str(version_path),
                num_rows=len(df),
                num_columns=len(df.columns),
                description=description or f"Version {next_ver}",
            )
            session.add(ver)

            # update the main dataset record to point to the latest version
            ds.file_path = str(version_path)
            ds.num_rows = len(df)
            ds.num_columns = len(df.columns)
            meta = DatasetIngestion.extract_metadata(df)
            ds.column_names = meta["column_names"]
            ds.column_types = meta["column_types"]
            ds.numerical_cols = meta["numerical_columns"]
            ds.categorical_cols = meta["categorical_columns"]
            ds.updated_at = datetime.now(timezone.utc)

            session.commit()
            logger.info("Created version %d for dataset %s", next_ver, dataset_id)
            return {
                "version": next_ver,
                "file_path": str(version_path),
                "num_rows": len(df),
                "num_columns": len(df.columns),
            }

        except DatasetNotFoundError:
            raise
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ── Update validation status ──────────────────────────────────────────

    @staticmethod
    def update_validation_status(dataset_id: str, status: str) -> None:
        """Update the validation_status field on a dataset."""
        session = get_session()
        try:
            ds = session.query(Dataset).filter_by(id=dataset_id).first()
            if ds is None:
                raise DatasetNotFoundError(f"Dataset '{dataset_id}' not found")
            ds.validation_status = status
            ds.updated_at = datetime.now(timezone.utc)
            session.commit()
        except DatasetNotFoundError:
            raise
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ── Delete ────────────────────────────────────────────────────────────

    @staticmethod
    def delete(dataset_id: str) -> None:
        """Delete a dataset and its files from disk."""
        session = get_session()
        try:
            ds = session.query(Dataset).filter_by(id=dataset_id).first()
            if ds is None:
                raise DatasetNotFoundError(f"Dataset '{dataset_id}' not found")

            # remove directory
            ds_dir = DATASETS_DIR / dataset_id
            if ds_dir.exists():
                shutil.rmtree(str(ds_dir), ignore_errors=True)

            session.delete(ds)
            session.commit()
            logger.info("Deleted dataset %s", dataset_id)
        except DatasetNotFoundError:
            raise
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ── Private helper ────────────────────────────────────────────────────

    @staticmethod
    def _to_dict(ds: Dataset) -> dict[str, Any]:
        return {
            "id": ds.id,
            "name": ds.name,
            "original_name": ds.original_name,
            "file_path": ds.file_path,
            "file_format": ds.file_format,
            "source": ds.source,
            "num_rows": ds.num_rows,
            "num_columns": ds.num_columns,
            "column_names": ds.column_names or [],
            "column_types": ds.column_types or {},
            "numerical_columns": ds.numerical_cols or [],
            "categorical_columns": ds.categorical_cols or [],
            "datetime_columns": ds.datetime_cols or [],
            "identifier_columns": ds.identifier_cols or [],
            "file_size_bytes": ds.file_size_bytes,
            "validation_status": ds.validation_status,
            "is_preprocessed": ds.is_preprocessed,
            "upload_time": str(ds.upload_time) if ds.upload_time else None,
            "updated_at": str(ds.updated_at) if ds.updated_at else None,
        }

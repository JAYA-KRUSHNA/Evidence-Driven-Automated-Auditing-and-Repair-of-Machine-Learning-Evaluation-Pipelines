"""
Dataset Loader — Download and load datasets from real sources.

Supports:
- scikit-learn built-in datasets
- OpenML datasets via the openml package
- UCI datasets via the ucimlrepo package
- User CSV/Parquet uploads
"""

import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np

from src.config import CACHE_DIR, SUPPORTED_UPLOAD_EXTENSIONS, MAX_UPLOAD_SIZE_MB
from src.data.dataset_cache import DatasetCache

logger = logging.getLogger(__name__)

# Initialize global cache
_cache = DatasetCache()


class DatasetLoadError(Exception):
    """Raised when a dataset cannot be loaded."""
    pass


class DatasetLoader:
    """
    Loads datasets from various sources with caching.
    All data returned is real — no fabrication.
    """

    def __init__(self):
        self.cache = _cache

    def load(self, source: str, source_id: str,
             name: str = "") -> Tuple[pd.DataFrame, dict]:
        """
        Load a dataset by source and ID.

        Returns:
            Tuple of (DataFrame, metadata_dict)

        Raises:
            DatasetLoadError if loading fails
        """
        # Check cache first
        cached_path = self.cache.get_cached_path(source, source_id)
        if cached_path is not None:
            logger.info(f"Loading cached dataset: {name or source_id}")
            df = pd.read_parquet(cached_path)
            metadata = {"source": source, "source_id": source_id, "cached": True}
            return df, metadata

        # Download from source
        if source == "sklearn":
            return self._load_sklearn(source_id, name)
        elif source == "openml":
            return self._load_openml(source_id, name)
        elif source == "uci":
            return self._load_uci(source_id, name)
        else:
            raise DatasetLoadError(f"Unknown data source: {source}")

    def load_from_file(self, file_path: Path,
                       name: str = "") -> Tuple[pd.DataFrame, dict]:
        """
        Load a dataset from a local file (CSV or Parquet).

        Returns:
            Tuple of (DataFrame, metadata_dict)
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise DatasetLoadError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_UPLOAD_EXTENSIONS:
            raise DatasetLoadError(
                f"Unsupported file type: {ext}. "
                f"Supported: {', '.join(SUPPORTED_UPLOAD_EXTENSIONS)}"
            )

        # Check file size
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_UPLOAD_SIZE_MB:
            raise DatasetLoadError(
                f"File too large: {size_mb:.1f} MB. Maximum: {MAX_UPLOAD_SIZE_MB} MB"
            )

        try:
            if ext == ".csv":
                df = pd.read_csv(file_path)
            elif ext == ".parquet":
                df = pd.read_parquet(file_path)
            else:
                raise DatasetLoadError(f"Unsupported format: {ext}")
        except Exception as e:
            raise DatasetLoadError(f"Failed to read file: {e}")

        if df.empty:
            raise DatasetLoadError("File is empty — no rows loaded.")

        if len(df.columns) < 2:
            raise DatasetLoadError(
                "File must have at least 2 columns (features + target)."
            )

        # Cache the uploaded file
        upload_name = name or file_path.stem
        source_id = self.cache.compute_checksum(file_path)[:12]

        # Copy to cache directory
        cache_path = CACHE_DIR / f"upload_{upload_name}.parquet"
        df.to_parquet(cache_path, index=False)
        self.cache.cache_file(cache_path, "upload", source_id, upload_name)

        metadata = {
            "source": "upload",
            "source_id": source_id,
            "original_file": str(file_path),
            "cached": False,
        }

        return df, metadata

    def load_from_upload(self, uploaded_file, name: str = "") -> Tuple[pd.DataFrame, dict]:
        """
        Load from a Streamlit UploadedFile object.
        """
        if uploaded_file is None:
            raise DatasetLoadError("No file provided.")

        # Save to temporary location first
        upload_name = name or uploaded_file.name.rsplit(".", 1)[0]
        ext = "." + uploaded_file.name.rsplit(".", 1)[-1].lower()

        if ext not in SUPPORTED_UPLOAD_EXTENSIONS:
            raise DatasetLoadError(
                f"Unsupported file type: {ext}. "
                f"Supported: {', '.join(SUPPORTED_UPLOAD_EXTENSIONS)}"
            )

        temp_path = CACHE_DIR / f"_temp_upload{ext}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            return self.load_from_file(temp_path, upload_name)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _load_sklearn(self, source_id: str,
                      name: str = "") -> Tuple[pd.DataFrame, dict]:
        """Load a scikit-learn built-in dataset."""
        try:
            from sklearn import datasets as sk_datasets

            loader_map = {
                "breast_cancer": sk_datasets.load_breast_cancer,
                "iris": sk_datasets.load_iris,
                "wine": sk_datasets.load_wine,
                "diabetes": sk_datasets.load_diabetes,
            }

            if source_id not in loader_map:
                raise DatasetLoadError(
                    f"Unknown sklearn dataset: {source_id}. "
                    f"Available: {', '.join(loader_map.keys())}"
                )

            data = loader_map[source_id](as_frame=True)
            df = data.frame

            # Cache it
            self.cache.cache_dataframe(df, "sklearn", source_id, name or source_id)

            metadata = {
                "source": "sklearn",
                "source_id": source_id,
                "description": data.DESCR[:500] if hasattr(data, 'DESCR') else "",
                "target_names": list(data.target_names) if hasattr(data, 'target_names') else [],
                "feature_names": list(data.feature_names) if hasattr(data, 'feature_names') else [],
                "cached": False,
            }

            return df, metadata

        except ImportError:
            raise DatasetLoadError("scikit-learn is not installed.")
        except Exception as e:
            raise DatasetLoadError(f"Failed to load sklearn dataset '{source_id}': {e}")

    def _load_openml(self, source_id: str,
                     name: str = "") -> Tuple[pd.DataFrame, dict]:
        """Load a dataset from OpenML."""
        try:
            import openml

            dataset_id = int(source_id)
            logger.info(f"Downloading OpenML dataset {dataset_id}...")

            dataset = openml.datasets.get_dataset(
                dataset_id, download_data=True,
                download_qualities=False, download_features_meta_data=False
            )

            df, targets, _, attribute_names = dataset.get_data(
                dataset_format="dataframe"
            )

            # If target is separate, add it
            if targets is not None and dataset.default_target_attribute:
                target_col = dataset.default_target_attribute
                if target_col not in df.columns:
                    df[target_col] = targets

            # Cache it
            self.cache.cache_dataframe(df, "openml", source_id, name or dataset.name)

            metadata = {
                "source": "openml",
                "source_id": source_id,
                "openml_name": dataset.name,
                "description": (dataset.description or "")[:500],
                "default_target": dataset.default_target_attribute,
                "url": f"https://www.openml.org/d/{dataset_id}",
                "cached": False,
            }

            return df, metadata

        except ImportError:
            raise DatasetLoadError("openml package is not installed. Run: pip install openml")
        except Exception as e:
            raise DatasetLoadError(f"Failed to load OpenML dataset {source_id}: {e}")

    def _load_uci(self, source_id: str,
                  name: str = "") -> Tuple[pd.DataFrame, dict]:
        """Load a dataset from UCI ML Repository."""
        try:
            from ucimlrepo import fetch_ucirepo

            dataset_id = int(source_id)
            logger.info(f"Downloading UCI dataset {dataset_id}...")

            result = fetch_ucirepo(id=dataset_id)

            # Use original data which includes all columns
            df = result.data.original
            if df is None:
                # Fall back to combining features and targets
                features_df = result.data.features
                targets_df = result.data.targets
                if features_df is not None and targets_df is not None:
                    df = pd.concat([features_df, targets_df], axis=1)
                elif features_df is not None:
                    df = features_df
                else:
                    raise DatasetLoadError(
                        f"UCI dataset {source_id} returned no data."
                    )

            # Cache it
            self.cache.cache_dataframe(df, "uci", source_id, name or result.metadata.name)

            # Extract target info
            target_col = ""
            if hasattr(result.metadata, 'target_col') and result.metadata.target_col:
                target_col = result.metadata.target_col
            elif result.data.targets is not None:
                target_col = result.data.targets.columns[0] if len(result.data.targets.columns) > 0 else ""

            metadata = {
                "source": "uci",
                "source_id": source_id,
                "uci_name": result.metadata.name if hasattr(result.metadata, 'name') else "",
                "description": str(getattr(result.metadata, 'additional_info', {}).get('summary', ''))[:500]
                if hasattr(result.metadata, 'additional_info') and isinstance(getattr(result.metadata, 'additional_info', None), dict)
                else "",
                "default_target": target_col,
                "url": f"https://archive.ics.uci.edu/dataset/{dataset_id}",
                "cached": False,
            }

            return df, metadata

        except ImportError:
            raise DatasetLoadError("ucimlrepo package is not installed. Run: pip install ucimlrepo")
        except Exception as e:
            raise DatasetLoadError(f"Failed to load UCI dataset {source_id}: {e}")

    def is_cached(self, source: str, source_id: str) -> bool:
        """Check if a dataset is already cached locally."""
        return self.cache.is_cached(source, source_id)

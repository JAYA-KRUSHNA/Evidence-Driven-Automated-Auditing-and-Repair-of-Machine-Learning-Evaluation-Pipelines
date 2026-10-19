"""
Dataset Cache — Local file caching with checksums.

Caches downloaded datasets to avoid repeated downloads.
Tracks file integrity via SHA-256 checksums.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import CACHE_DIR


class DatasetCache:
    """Manages local caching of downloaded datasets."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / "manifest.json"
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        """Load the cache manifest from disk."""
        if self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                return json.load(f)
        return {}

    def _save_manifest(self):
        """Save the cache manifest to disk."""
        with open(self.manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

    @staticmethod
    def compute_checksum(file_path: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def compute_df_checksum(df: pd.DataFrame) -> str:
        """Compute checksum of a DataFrame's content."""
        return hashlib.sha256(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()

    def get_cache_key(self, source: str, source_id: str) -> str:
        """Generate a unique cache key."""
        return f"{source}_{source_id}"

    def get_cached_path(self, source: str, source_id: str) -> Optional[Path]:
        """Get the cached file path if it exists and is valid."""
        key = self.get_cache_key(source, source_id)
        if key in self._manifest:
            cached_path = Path(self._manifest[key]["file_path"])
            if cached_path.exists():
                # Verify checksum
                stored_checksum = self._manifest[key].get("checksum")
                if stored_checksum:
                    actual_checksum = self.compute_checksum(cached_path)
                    if actual_checksum == stored_checksum:
                        return cached_path
                else:
                    return cached_path
        return None

    def cache_dataframe(self, df: pd.DataFrame, source: str,
                        source_id: str, name: str) -> Path:
        """Cache a DataFrame as a parquet file."""
        key = self.get_cache_key(source, source_id)
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        file_path = self.cache_dir / f"{safe_name}_{source_id}.parquet"

        df.to_parquet(file_path, index=False)
        checksum = self.compute_checksum(file_path)

        self._manifest[key] = {
            "file_path": str(file_path),
            "name": name,
            "source": source,
            "source_id": source_id,
            "checksum": checksum,
            "rows": len(df),
            "cols": len(df.columns),
        }
        self._save_manifest()
        return file_path

    def cache_file(self, file_path: Path, source: str,
                   source_id: str, name: str) -> Path:
        """Register an existing file in the cache."""
        key = self.get_cache_key(source, source_id)
        checksum = self.compute_checksum(file_path)

        self._manifest[key] = {
            "file_path": str(file_path),
            "name": name,
            "source": source,
            "source_id": source_id,
            "checksum": checksum,
        }
        self._save_manifest()
        return file_path

    def is_cached(self, source: str, source_id: str) -> bool:
        """Check if a dataset is cached."""
        return self.get_cached_path(source, source_id) is not None

    def invalidate(self, source: str, source_id: str):
        """Remove a dataset from cache."""
        key = self.get_cache_key(source, source_id)
        if key in self._manifest:
            cached_path = Path(self._manifest[key]["file_path"])
            if cached_path.exists():
                cached_path.unlink()
            del self._manifest[key]
            self._save_manifest()

    def get_cache_info(self) -> dict:
        """Get cache statistics."""
        total_size = 0
        cached_count = 0
        for key, info in self._manifest.items():
            path = Path(info["file_path"])
            if path.exists():
                total_size += path.stat().st_size
                cached_count += 1
        return {
            "total_entries": len(self._manifest),
            "valid_entries": cached_count,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }

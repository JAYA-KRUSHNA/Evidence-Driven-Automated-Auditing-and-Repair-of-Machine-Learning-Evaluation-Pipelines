"""
Datasets REST API endpoints.
"""

import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from ..config import DATASETS_DIR, MAX_UPLOAD_SIZE_MB
from ..dataset.ingestion import DatasetIngestion, DatasetReadError
from ..dataset.registry import DatasetNotFoundError, DatasetRegistry
from ..dataset.samples import SampleDatasetManager

router = APIRouter(prefix="/api/datasets", tags=["Datasets"])


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
):
    """Upload a CSV or Excel dataset file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".csv", ".xlsx", ".xls"):
        raise HTTPException(status_code=400, detail=f"Unsupported format '{suffix}'. Supported: .csv, .xlsx, .xls")

    tmp_dir = DATASETS_DIR / "_temp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = tmp_dir / file.filename

    try:
        with open(tmp_file, "wb") as f:
            shutil.copyfileobj(file.file, f)

        size_mb = tmp_file.stat().st_size / (1024 * 1024)
        if size_mb > MAX_UPLOAD_SIZE_MB:
            raise HTTPException(status_code=413, detail=f"File exceeds maximum size of {MAX_UPLOAD_SIZE_MB}MB")

        df, metadata = DatasetIngestion.read(tmp_file)

        record = DatasetRegistry.register(
            df=df,
            metadata=metadata,
            name=name,
            source="upload",
            original_file_path=tmp_file,
        )
        return record
    except DatasetReadError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process dataset: {e}")
    finally:
        if tmp_file.exists():
            tmp_file.unlink()


@router.get("", status_code=status.HTTP_200_OK)
def list_datasets():
    """List all registered datasets."""
    return DatasetRegistry.list_all()


@router.get("/samples/available", status_code=status.HTTP_200_OK)
def list_sample_datasets():
    """List all built-in sample datasets."""
    return SampleDatasetManager.list_samples()


@router.post("/sample/{sample_name}", status_code=status.HTTP_201_CREATED)
def load_sample_dataset(sample_name: str):
    """Load a built-in sample dataset into the registry."""
    # Friendly alias handling
    clean_name = sample_name.lower().strip()
    if clean_name == "diabetes":
        clean_name = "diabetes_regression"

    try:
        df, meta = SampleDatasetManager.load_sample(clean_name)
        record = DatasetRegistry.register(
            df=df,
            metadata=meta,
            name=meta.get("display_name", clean_name.title()),
            source="sample",
        )
        return record
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}", status_code=status.HTTP_200_OK)
def get_dataset(dataset_id: str, preview_rows: int = 10):
    """Get dataset metadata and preview rows."""
    try:
        meta = DatasetRegistry.get(dataset_id)
        df = DatasetRegistry.load_dataframe(dataset_id)
        preview = df.head(preview_rows).replace({float("nan"): None}).to_dict(orient="records")
        return {
            "metadata": meta,
            "preview": preview,
            "columns": list(df.columns),
            "shape": {"rows": len(df), "columns": len(df.columns)},
        }
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{dataset_id}", status_code=status.HTTP_200_OK)
def delete_dataset(dataset_id: str):
    """Delete a dataset and its versions from registry and disk."""
    try:
        DatasetRegistry.delete(dataset_id)
        return {"status": "success", "message": f"Dataset '{dataset_id}' deleted"}
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/download")
def download_dataset(dataset_id: str):
    """Download the underlying dataset CSV/Excel file."""
    try:
        meta = DatasetRegistry.get(dataset_id)
        file_path = Path(meta["file_path"])
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Dataset file missing on disk")
        return FileResponse(
            path=file_path,
            filename=meta.get("original_name", f"{dataset_id}.csv"),
            media_type="application/octet-stream",
        )
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

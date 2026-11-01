"""
Models REST API endpoints.
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse

from ..models.registry import ModelNotFoundError, ModelRegistry

router = APIRouter(prefix="/api/models", tags=["Models"])


@router.get("", status_code=status.HTTP_200_OK)
def list_models(
    dataset_id: Optional[str] = Query(None),
    experiment_id: Optional[str] = Query(None),
    task_type: Optional[str] = Query(None),
    algorithm: Optional[str] = Query(None),
):
    """List trained models with optional filters."""
    return ModelRegistry.list_all(
        dataset_id=dataset_id,
        experiment_id=experiment_id,
        task_type=task_type,
        algorithm=algorithm,
    )


@router.get("/{model_id}", status_code=status.HTTP_200_OK)
def get_model(model_id: str):
    """Retrieve metadata and files for a trained model."""
    try:
        return ModelRegistry.get(model_id)
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


@router.delete("/{model_id}", status_code=status.HTTP_200_OK)
def delete_model(model_id: str):
    """Delete a trained model and remove its disk artifacts."""
    try:
        ModelRegistry.delete(model_id)
        return {"status": "success", "message": f"Model '{model_id}' deleted"}
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_id}/download")
def download_model(model_id: str, format: str = Query("joblib", pattern="^(joblib|zip)$")):
    """Download serialized model file (joblib) or full bundle (zip)."""
    try:
        export_path = ModelRegistry.export(model_id, format=format)
        path = Path(export_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Model file missing on disk")

        filename = f"{model_id}.{format}"
        media_type = "application/zip" if format == "zip" else "application/octet-stream"
        return FileResponse(path=path, filename=filename, media_type=media_type)
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_id}/best", status_code=status.HTTP_200_OK)
def set_best_model(model_id: str):
    """Mark a model as the best model for its experiment."""
    try:
        ModelRegistry.set_best(model_id)
        return {"status": "success", "message": f"Model '{model_id}' marked as best"}
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

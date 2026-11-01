"""
Experiments REST API endpoints.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from ..experiment.tracker import ExperimentNotFoundError, ExperimentTracker

router = APIRouter(prefix="/api/experiments", tags=["Experiments"])


class ExperimentCompareRequest(BaseModel):
    experiment_ids: list[str]


@router.get("", status_code=status.HTTP_200_OK)
def list_experiments(
    dataset_id: Optional[str] = Query(None),
    task_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
):
    """List tracked experiments with optional filtering."""
    return ExperimentTracker.list_all(
        dataset_id=dataset_id,
        task_type=task_type,
        status=status,
    )


@router.get("/{experiment_id}", status_code=status.HTTP_200_OK)
def get_experiment(experiment_id: str):
    """Retrieve full experiment details including evaluations and models."""
    try:
        return ExperimentTracker.get(experiment_id)
    except ExperimentNotFoundError:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")


@router.delete("/{experiment_id}", status_code=status.HTTP_200_OK)
def delete_experiment(experiment_id: str):
    """Delete an experiment and all linked records."""
    try:
        ExperimentTracker.delete(experiment_id)
        return {"status": "success", "message": f"Experiment '{experiment_id}' deleted"}
    except ExperimentNotFoundError:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", status_code=status.HTTP_200_OK)
def compare_experiments(req: ExperimentCompareRequest):
    """Compare multiple experiments side by side on evaluation metrics."""
    return ExperimentTracker.compare(req.experiment_ids)

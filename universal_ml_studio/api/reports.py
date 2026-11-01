"""
Reports REST API endpoints.
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..reports.generator import ReportGenerator, ReportNotFoundError

router = APIRouter(prefix="/api/reports", tags=["Reports"])


class GenerateReportRequest(BaseModel):
    report_type: str  # 'validation' | 'eda' | 'experiment' | 'full'
    format: str = "json"  # 'json' | 'csv' | 'pdf'
    dataset_id: Optional[str] = None
    experiment_id: Optional[str] = None


@router.post("/generate", status_code=status.HTTP_201_CREATED)
def generate_report(req: GenerateReportRequest):
    """Generate and store an audit report in JSON, CSV, or PDF format."""
    rtype = req.report_type.lower()
    fmt = req.format.lower()

    try:
        if rtype == "validation":
            if not req.dataset_id:
                raise HTTPException(status_code=400, detail="dataset_id is required for validation reports")
            return ReportGenerator.generate_validation_report(req.dataset_id, format=fmt)

        elif rtype == "eda":
            if not req.dataset_id:
                raise HTTPException(status_code=400, detail="dataset_id is required for eda reports")
            return ReportGenerator.generate_eda_report(req.dataset_id, format=fmt)

        elif rtype == "experiment":
            if not req.experiment_id:
                raise HTTPException(status_code=400, detail="experiment_id is required for experiment reports")
            return ReportGenerator.generate_experiment_report(req.experiment_id, format=fmt)

        elif rtype == "full":
            if not req.dataset_id:
                raise HTTPException(status_code=400, detail="dataset_id is required for full reports")
            return ReportGenerator.generate_full_report(req.dataset_id, format=fmt)

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported report_type '{req.report_type}'. Choose from: validation, eda, experiment, full",
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", status_code=status.HTTP_200_OK)
def list_reports(
    dataset_id: Optional[str] = Query(None),
    experiment_id: Optional[str] = Query(None),
    report_type: Optional[str] = Query(None),
):
    """List generated reports with optional filtering."""
    return ReportGenerator.list_all(
        dataset_id=dataset_id,
        experiment_id=experiment_id,
        report_type=report_type,
    )


@router.get("/{report_id}", status_code=status.HTTP_200_OK)
def get_report(report_id: str):
    """Retrieve metadata for a generated report."""
    try:
        return ReportGenerator.get(report_id)
    except ReportNotFoundError:
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")


@router.delete("/{report_id}", status_code=status.HTTP_200_OK)
def delete_report(report_id: str):
    """Delete a report record and remove its file from disk."""
    try:
        ReportGenerator.delete(report_id)
        return {"status": "success", "message": f"Report '{report_id}' deleted"}
    except ReportNotFoundError:
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{report_id}/download")
def download_report(report_id: str):
    """Download the generated report file."""
    try:
        rep = ReportGenerator.get(report_id)
        path = Path(rep["file_path"])
        if not path.exists():
            raise HTTPException(status_code=404, detail="Report file missing on disk")

        fmt = rep.get("format", "json")
        media_types = {
            "pdf": "application/pdf",
            "csv": "text/csv",
            "json": "application/json",
        }
        return FileResponse(
            path=path,
            filename=path.name,
            media_type=media_types.get(fmt, "application/octet-stream"),
        )
    except ReportNotFoundError:
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")

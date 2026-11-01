"""
Validation REST API endpoints.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status

from ..database.connection import get_session
from ..database.models import ValidationReport
from ..dataset.registry import DatasetNotFoundError, DatasetRegistry
from ..validation.engine import ValidationEngine

router = APIRouter(prefix="/api/validation", tags=["Validation"])


@router.post("/run/{dataset_id}", status_code=status.HTTP_200_OK)
def run_validation(
    dataset_id: str,
    target_column: Optional[str] = Query(None, description="Optional target column for ML readiness checks"),
):
    """Run full 4-tier validation suite on a dataset."""
    try:
        ds_meta = DatasetRegistry.get(dataset_id)
        df = DatasetRegistry.load_dataframe(dataset_id)

        meta = {
            "file_name": ds_meta.get("original_name", "dataset.csv"),
            "file_format": ds_meta.get("file_format", "csv"),
            "file_size_bytes": ds_meta.get("file_size_bytes", 0),
        }

        result = ValidationEngine.validate(df, meta, target_column=target_column)
        report_row = ValidationEngine.save_report(result, dataset_id)

        # Update dataset validation status
        DatasetRegistry.update_validation_status(dataset_id, result.status)

        out = result.to_dict()
        out["report_id"] = report_row.id if report_row else None
        return out
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}", status_code=status.HTTP_200_OK)
def get_validation_report(dataset_id: str):
    """Retrieve the latest validation report for a dataset."""
    session = get_session()
    try:
        # Check dataset exists
        try:
            DatasetRegistry.get(dataset_id)
        except DatasetNotFoundError:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

        report = (
            session.query(ValidationReport)
            .filter_by(dataset_id=dataset_id)
            .order_by(ValidationReport.created_at.desc())
            .first()
        )
        if not report:
            raise HTTPException(status_code=404, detail=f"No validation report found for dataset '{dataset_id}'")

        checks = [
            {
                "id": c.id,
                "tier": c.category,
                "check_name": c.check_name,
                "status": c.status,
                "message": c.message,
                "affected_columns": c.affected_columns,
                "affected_rows": c.affected_rows,
                "suggestion": c.suggestion,
                "details": c.details,
            }
            for c in report.checks
        ]

        return {
            "id": report.id,
            "dataset_id": report.dataset_id,
            "status": report.status,
            "total_checks": report.total_checks,
            "passed_checks": report.passed_checks,
            "failed_checks": report.failed_checks,
            "warning_count": report.warning_count,
            "is_trainable": report.is_trainable,
            "summary": report.summary,
            "created_at": report.created_at.isoformat() if report.created_at else None,
            "checks": checks,
        }
    finally:
        session.close()

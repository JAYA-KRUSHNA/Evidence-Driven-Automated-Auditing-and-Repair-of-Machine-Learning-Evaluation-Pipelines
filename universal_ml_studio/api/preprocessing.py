"""
Preprocessing REST API endpoints.
"""

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from ..database.connection import get_session
from ..database.models import PreprocessingHistory
from ..dataset.registry import DatasetNotFoundError, DatasetRegistry
from ..preprocessing.engine import PreprocessingEngine

router = APIRouter(prefix="/api/preprocessing", tags=["Preprocessing"])


class AutoPreprocessRequest(BaseModel):
    target_column: Optional[str] = None


class OperationRequest(BaseModel):
    operation_type: str  # 'impute' | 'encode' | 'scale' | 'outliers' | 'datetime'
    strategy: Optional[str] = None
    columns: Optional[list[str]] = None


@router.post("/auto/{dataset_id}", status_code=status.HTTP_200_OK)
def auto_preprocess(dataset_id: str, payload: Optional[AutoPreprocessRequest] = None):
    """Run automated full preprocessing pipeline on a dataset and save new version."""
    try:
        meta = DatasetRegistry.get(dataset_id)
        df = DatasetRegistry.load_dataframe(dataset_id)
        target_col = payload.target_column if payload else None

        df_clean, operations = PreprocessingEngine.auto_preprocess(
            df=df,
            metadata=meta,
            target_column=target_col,
        )

        # Save history in DB
        PreprocessingEngine.save_history(operations, dataset_id)

        # Create new dataset version
        version = DatasetRegistry.create_version(
            dataset_id=dataset_id,
            df=df_clean,
            description="Automated preprocessing pipeline",
        )

        return {
            "status": "success",
            "dataset_id": dataset_id,
            "version": version.get("version", 1),
            "operations_applied": len(operations),
            "operations": operations,
            "shape_before": {"rows": meta["num_rows"], "columns": meta["num_columns"]},
            "shape_after": {"rows": len(df_clean), "columns": len(df_clean.columns)},
        }
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/operation/{dataset_id}", status_code=status.HTTP_200_OK)
def apply_operation(dataset_id: str, op: OperationRequest):
    """Apply a specific preprocessing step (impute, encode, scale, outliers, datetime)."""
    try:
        meta = DatasetRegistry.get(dataset_id)
        df = DatasetRegistry.load_dataframe(dataset_id)

        op_type = op.operation_type.lower()
        if op_type == "impute":
            strategy = op.strategy or "mean"
            df_out, log = PreprocessingEngine.impute_missing(df, strategy=strategy, columns=op.columns)
        elif op_type == "encode":
            method = op.strategy or "label"
            df_out, log = PreprocessingEngine.encode_categorical(df, method=method, columns=op.columns)
        elif op_type == "scale":
            method = op.strategy or "standard"
            df_out, log = PreprocessingEngine.scale_features(df, method=method, columns=op.columns)
        elif op_type == "outliers":
            method = op.strategy or "iqr_cap"
            df_out, log = PreprocessingEngine.handle_outliers(df, method=method, columns=op.columns)
        elif op_type == "datetime":
            df_out, log = PreprocessingEngine.extract_datetime_features(df, columns=op.columns)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported operation_type '{op.operation_type}'. Use impute, encode, scale, outliers, datetime.",
            )

        PreprocessingEngine.save_history([log], dataset_id)
        version = DatasetRegistry.create_version(
            dataset_id=dataset_id,
            df=df_out,
            description=f"Preprocessing: {op_type} ({op.strategy or 'default'})",
        )

        return {
            "status": "success",
            "dataset_id": dataset_id,
            "version": version.get("version", 1),
            "operation": log,
            "shape_after": {"rows": len(df_out), "columns": len(df_out.columns)},
        }
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{dataset_id}", status_code=status.HTTP_200_OK)
def get_preprocessing_history(dataset_id: str):
    """Retrieve full history of preprocessing transformations applied to a dataset."""
    session = get_session()
    try:
        try:
            DatasetRegistry.get(dataset_id)
        except DatasetNotFoundError:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

        history = (
            session.query(PreprocessingHistory)
            .filter_by(dataset_id=dataset_id)
            .order_by(PreprocessingHistory.applied_at.asc())
            .all()
        )

        return [
            {
                "id": h.id,
                "dataset_id": h.dataset_id,
                "operation": h.operation,
                "category": h.category,
                "parameters": h.parameters or {},
                "affected_columns": h.affected_columns or [],
                "rows_before": h.rows_before,
                "rows_after": h.rows_after,
                "applied_at": h.applied_at.isoformat() if h.applied_at else None,
            }
            for h in history
        ]
    finally:
        session.close()

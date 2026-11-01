"""
EDA REST API endpoints.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status

from ..database.connection import get_session
from ..database.models import EDAResult
from ..dataset.registry import DatasetNotFoundError, DatasetRegistry
from ..eda.engine import EDAEngine

router = APIRouter(prefix="/api/eda", tags=["Exploratory Data Analysis"])


@router.post("/run/{dataset_id}", status_code=status.HTTP_200_OK)
def run_eda(
    dataset_id: str,
    target_column: Optional[str] = Query(None, description="Optional target column for target analysis"),
):
    """Execute full automated exploratory data analysis on a dataset."""
    try:
        ds_meta = DatasetRegistry.get(dataset_id)
        df = DatasetRegistry.load_dataframe(dataset_id)

        meta = {
            "file_name": ds_meta.get("original_name", "dataset.csv"),
            "file_format": ds_meta.get("file_format", "csv"),
        }

        eda_payload = EDAEngine.analyze(df, meta, target_column=target_column)
        EDAEngine.save(eda_payload, dataset_id)

        return {
            "dataset_id": dataset_id,
            "status": "success",
            "modules_run": list(eda_payload.keys()),
            "eda": eda_payload,
        }
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}", status_code=status.HTTP_200_OK)
def get_eda(dataset_id: str):
    """Retrieve existing EDA results for a dataset (or runs it if not yet computed)."""
    session = get_session()
    try:
        try:
            ds_meta = DatasetRegistry.get(dataset_id)
        except DatasetNotFoundError:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

        records = session.query(EDAResult).filter_by(dataset_id=dataset_id).all()
        if records:
            eda_payload = {r.analysis_type: r.result_data for r in records}
        else:
            df = DatasetRegistry.load_dataframe(dataset_id)
            meta = {
                "file_name": ds_meta.get("original_name", "dataset.csv"),
                "file_format": ds_meta.get("file_format", "csv"),
            }
            eda_payload = EDAEngine.analyze(df, meta)
            EDAEngine.save(eda_payload, dataset_id, session=session)

        return {
            "dataset_id": dataset_id,
            "modules": list(eda_payload.keys()),
            "eda": eda_payload,
        }
    finally:
        session.close()


@router.get("/{dataset_id}/{module_name}", status_code=status.HTTP_200_OK)
def get_eda_module(dataset_id: str, module_name: str):
    """Retrieve a specific EDA module (e.g. overview, statistics, correlation, missing_analysis)."""
    session = get_session()
    try:
        try:
            DatasetRegistry.get(dataset_id)
        except DatasetNotFoundError:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

        record = (
            session.query(EDAResult)
            .filter_by(dataset_id=dataset_id, analysis_type=module_name)
            .first()
        )
        if not record:
            # If not yet computed, compute full and return module
            df = DatasetRegistry.load_dataframe(dataset_id)
            meta = DatasetRegistry.get(dataset_id)
            full_eda = EDAEngine.analyze(df, meta)
            EDAEngine.save(full_eda, dataset_id, session=session)
            if module_name not in full_eda:
                raise HTTPException(status_code=404, detail=f"Module '{module_name}' not available")
            return {"dataset_id": dataset_id, "module": module_name, "data": full_eda[module_name]}

        return {"dataset_id": dataset_id, "module": module_name, "data": record.result_data}
    finally:
        session.close()

"""
EDA REST API endpoints.
Provides complete automated exploratory data analysis, interactive chart generation,
and multi-format report exports (PDF, HTML, Excel).
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Response, status
from fastapi.responses import Response

from ..database.connection import get_session
from ..database.models import EDAResult
from ..dataset.registry import DatasetNotFoundError, DatasetRegistry
from ..eda.engine import EDAEngine

router = APIRouter(prefix="/api/eda", tags=["Exploratory Data Analysis"])


@router.post("/run/{dataset_id}", status_code=status.HTTP_200_OK)
def run_eda(
    dataset_id: str,
    target_column: Optional[str] = Query(None, description="Optional target column for target analysis"),
    task_type: Optional[str] = Query(None, description="Optional task type override: 'classification' or 'regression'"),
):
    """Execute full automated exploratory data analysis on a dataset."""
    try:
        ds_meta = DatasetRegistry.get(dataset_id)
        df = DatasetRegistry.load_dataframe(dataset_id)

        meta = {
            "file_name": ds_meta.get("original_name", "dataset.csv"),
            "file_format": ds_meta.get("file_format", "csv"),
            "numerical_columns": ds_meta.get("numerical_columns", []),
            "categorical_columns": ds_meta.get("categorical_columns", []),
            "datetime_columns": ds_meta.get("datetime_columns", []),
            "column_types": ds_meta.get("column_types", {}),
        }

        eda_payload = EDAEngine.analyze(df, meta, target_column=target_column, task_type=task_type)
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
    """Retrieve existing EDA results for a dataset (or compute if not yet stored)."""
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
                "numerical_columns": ds_meta.get("numerical_columns", []),
                "categorical_columns": ds_meta.get("categorical_columns", []),
                "datetime_columns": ds_meta.get("datetime_columns", []),
                "column_types": ds_meta.get("column_types", {}),
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


@router.get("/{dataset_id}/export", status_code=status.HTTP_200_OK)
def export_eda_report(
    dataset_id: str,
    format: str = Query("html", description="Report format: 'html', 'pdf', or 'excel'"),
    target_column: Optional[str] = Query(None, description="Optional target column"),
):
    """Export complete EDA audit report as downloadable PDF, HTML, or Excel file."""
    try:
        ds_meta = DatasetRegistry.get(dataset_id)
        df = DatasetRegistry.load_dataframe(dataset_id)
        dataset_name = ds_meta.get("name") or ds_meta.get("original_name", "Dataset")

        meta = {
            "file_name": ds_meta.get("original_name", "dataset.csv"),
            "file_format": ds_meta.get("file_format", "csv"),
            "numerical_columns": ds_meta.get("numerical_columns", []),
            "categorical_columns": ds_meta.get("categorical_columns", []),
            "datetime_columns": ds_meta.get("datetime_columns", []),
            "column_types": ds_meta.get("column_types", {}),
        }
        eda_data = EDAEngine.analyze(df, meta, target_column=target_column)

        content, media_type, filename = EDAEngine.generate_report(dataset_name, eda_data, format=format)
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/chart/{chart_type}", status_code=status.HTTP_200_OK)
def get_interactive_chart(
    dataset_id: str,
    chart_type: str,
    col1: Optional[str] = Query(None, description="Primary column name"),
    col2: Optional[str] = Query(None, description="Secondary column name"),
    column: Optional[str] = Query(None, description="Alternative column alias"),
):
    """Generate dynamic Plotly interactive figure specification."""
    try:
        DatasetRegistry.get(dataset_id)
        df = DatasetRegistry.load_dataframe(dataset_id)

        params = {"col1": col1 or column, "col2": col2, "column": column or col1}
        chart_dict = EDAEngine.generate_chart(df, chart_type, params)

        if "error" in chart_dict:
            raise HTTPException(status_code=400, detail=chart_dict["error"])

        return {
            "dataset_id": dataset_id,
            "chart_type": chart_type,
            "params": params,
            "figure": chart_dict,
        }
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/univariate/detail", status_code=status.HTTP_200_OK)
def get_univariate_column(
    dataset_id: str,
    column: str = Query(..., description="Target column to analyze"),
):
    """Calculate granular univariate statistics and distribution for a specific column."""
    try:
        ds_meta = DatasetRegistry.get(dataset_id)
        df = DatasetRegistry.load_dataframe(dataset_id)
        if column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column}' not found in dataset")

        res = EDAEngine.get_univariate(df, ds_meta, column=column)
        return {"dataset_id": dataset_id, "result": res}
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/bivariate/pair", status_code=status.HTTP_200_OK)
def get_bivariate_pair(
    dataset_id: str,
    col1: str = Query(..., description="First column"),
    col2: str = Query(..., description="Second column"),
):
    """Calculate bivariate relationship between two specific columns."""
    try:
        ds_meta = DatasetRegistry.get(dataset_id)
        df = DatasetRegistry.load_dataframe(dataset_id)
        if col1 not in df.columns or col2 not in df.columns:
            raise HTTPException(status_code=400, detail=f"Columns '{col1}' or '{col2}' not found")

        res = EDAEngine.get_bivariate(df, ds_meta, col1=col1, col2=col2)
        return {"dataset_id": dataset_id, "result": res}
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/outliers/method", status_code=status.HTTP_200_OK)
def get_outliers_by_method(
    dataset_id: str,
    method: str = Query("all", description="Detection method: 'iqr', 'zscore', 'modified_zscore', 'isolation_forest', 'lof', or 'all'"),
):
    """Run outlier detection using a specific selected method."""
    try:
        ds_meta = DatasetRegistry.get(dataset_id)
        df = DatasetRegistry.load_dataframe(dataset_id)
        res = EDAEngine.get_outliers(df, ds_meta, method=method)
        return {"dataset_id": dataset_id, "method": method, "result": res}
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/{module_name}", status_code=status.HTTP_200_OK)
def get_eda_module(dataset_id: str, module_name: str):
    """Retrieve a specific EDA module (e.g. overview, quality, statistics, correlation, missing_analysis, multivariate, features, insights)."""
    session = get_session()
    try:
        try:
            ds_meta = DatasetRegistry.get(dataset_id)
        except DatasetNotFoundError:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

        record = (
            session.query(EDAResult)
            .filter_by(dataset_id=dataset_id, analysis_type=module_name)
            .first()
        )
        if not record:
            df = DatasetRegistry.load_dataframe(dataset_id)
            meta = {
                "file_name": ds_meta.get("original_name", "dataset.csv"),
                "file_format": ds_meta.get("file_format", "csv"),
                "numerical_columns": ds_meta.get("numerical_columns", []),
                "categorical_columns": ds_meta.get("categorical_columns", []),
                "datetime_columns": ds_meta.get("datetime_columns", []),
                "column_types": ds_meta.get("column_types", {}),
            }
            full_eda = EDAEngine.analyze(df, meta)
            EDAEngine.save(full_eda, dataset_id, session=session)
            if module_name not in full_eda:
                raise HTTPException(status_code=404, detail=f"Module '{module_name}' not available")
            return {"dataset_id": dataset_id, "module": module_name, "data": full_eda[module_name]}

        return {"dataset_id": dataset_id, "module": module_name, "data": record.result_data}
    finally:
        session.close()

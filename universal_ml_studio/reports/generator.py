"""
Report Generator — produces PDF, CSV, and JSON audit reports for datasets,
validations, EDA, and ML experiments.

Public API
----------
    ReportGenerator.generate_validation_report(dataset_id, format='json') -> dict
    ReportGenerator.generate_eda_report(dataset_id, format='json')        -> dict
    ReportGenerator.generate_experiment_report(exp_id, format='json')     -> dict
    ReportGenerator.generate_full_report(dataset_id, format='json')       -> dict
    ReportGenerator.get(report_id)                                        -> dict
    ReportGenerator.list_all(...)                                         -> list[dict]
    ReportGenerator.delete(report_id)                                     -> bool
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from ..config import REPORTS_DIR
from ..database.connection import get_session
from ..database.models import Dataset, EDAResult, Report, ValidationReport
from ..dataset.registry import DatasetRegistry
from ..eda.engine import EDAEngine
from ..experiment.tracker import ExperimentTracker
from ..validation.engine import ValidationEngine

logger = logging.getLogger(__name__)


class ReportNotFoundError(Exception):
    """Raised when a report ID cannot be found."""


def _default_serializer(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, (np.ndarray, set)):
        return list(obj)
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    return str(obj)


class ReportGenerator:
    """Generates and archives reports in JSON, CSV, and PDF formats."""

    @classmethod
    def generate_validation_report(
        cls,
        dataset_id: str,
        format: str = "json",
    ) -> dict[str, Any]:
        """Generate a validation audit report."""
        cls._validate_format(format)
        session = get_session()
        try:
            ds = session.query(Dataset).filter_by(id=dataset_id).first()
            if not ds:
                raise ValueError(f"Dataset '{dataset_id}' not found.")

            # Get latest validation report or run validation
            val_rep = (
                session.query(ValidationReport)
                .filter_by(dataset_id=dataset_id)
                .order_by(ValidationReport.created_at.desc())
                .first()
            )

            if val_rep:
                val_data = {
                    "dataset_id": dataset_id,
                    "dataset_name": ds.name,
                    "generated_at": val_rep.created_at.isoformat() if val_rep.created_at else None,
                    "status": val_rep.status,
                    "is_trainable": val_rep.is_trainable,
                    "total_checks": val_rep.total_checks,
                    "passed_checks": val_rep.passed_checks,
                    "failed_checks": val_rep.failed_checks,
                    "warning_count": val_rep.warning_count,
                    "summary": val_rep.summary or {},
                    "checks": [
                        {
                            "tier": c.category,
                            "check_name": c.check_name,
                            "status": c.status,
                            "message": c.message,
                            "details": c.details,
                        }
                        for c in val_rep.checks
                    ],
                }
            else:
                df = DatasetRegistry.load_dataframe(dataset_id)
                v_res = ValidationEngine.validate(df, {"file_name": ds.original_name, "file_format": ds.file_format})
                val_data = v_res.to_dict()
                val_data["dataset_id"] = dataset_id
                val_data["dataset_name"] = ds.name
                val_data["generated_at"] = datetime.utcnow().isoformat()

            report_id = uuid.uuid4().hex[:12]
            file_name = f"val_{dataset_id}_{report_id}.{format}"
            out_path = REPORTS_DIR / file_name

            if format == "json":
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(val_data, f, indent=2, default=_default_serializer)

            elif format == "csv":
                checks = val_data.get("checks", [])
                df_checks = pd.DataFrame(checks)
                if df_checks.empty:
                    df_checks = pd.DataFrame([{"summary": val_data.get("summary", "")}])
                df_checks.to_csv(out_path, index=False)

            elif format == "pdf":
                cls._build_validation_pdf(out_path, val_data)

            file_size = out_path.stat().st_size
            report_rec = Report(
                id=report_id,
                dataset_id=dataset_id,
                report_type="validation",
                format=format,
                file_path=str(out_path),
                file_size=file_size,
            )
            session.add(report_rec)
            session.commit()

            return cls._to_dict(report_rec)
        finally:
            session.close()

    @classmethod
    def generate_eda_report(
        cls,
        dataset_id: str,
        format: str = "json",
    ) -> dict[str, Any]:
        """Generate an exploratory data analysis report."""
        cls._validate_format(format)
        session = get_session()
        try:
            ds = session.query(Dataset).filter_by(id=dataset_id).first()
            if not ds:
                raise ValueError(f"Dataset '{dataset_id}' not found.")

            eda_records = session.query(EDAResult).filter_by(dataset_id=dataset_id).all()
            if eda_records:
                eda_payload = {r.analysis_type: r.result_data for r in eda_records}
            else:
                df = DatasetRegistry.load_dataframe(dataset_id)
                eda_payload = EDAEngine.analyze(df, {"file_name": ds.original_name, "file_format": ds.file_format})

            eda_data = {
                "dataset_id": dataset_id,
                "dataset_name": ds.name,
                "rows": ds.num_rows,
                "columns": ds.num_columns,
                "eda": eda_payload,
            }

            report_id = uuid.uuid4().hex[:12]
            file_name = f"eda_{dataset_id}_{report_id}.{format}"
            out_path = REPORTS_DIR / file_name

            if format == "json":
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(eda_data, f, indent=2, default=_default_serializer)

            elif format == "csv":
                # Output statistics table or missing analysis
                stats = eda_payload.get("statistics", {})
                num_stats = stats.get("numerical", {})
                if num_stats:
                    df_stats = pd.DataFrame.from_dict(num_stats, orient="index")
                    df_stats.index.name = "column"
                    df_stats.reset_index(inplace=True)
                else:
                    df_stats = pd.DataFrame([{"info": "No numerical statistics available"}])
                df_stats.to_csv(out_path, index=False)

            elif format == "pdf":
                cls._build_eda_pdf(out_path, eda_data)

            file_size = out_path.stat().st_size
            report_rec = Report(
                id=report_id,
                dataset_id=dataset_id,
                report_type="eda",
                format=format,
                file_path=str(out_path),
                file_size=file_size,
            )
            session.add(report_rec)
            session.commit()

            return cls._to_dict(report_rec)
        finally:
            session.close()

    @classmethod
    def generate_experiment_report(
        cls,
        experiment_id: str,
        format: str = "json",
    ) -> dict[str, Any]:
        """Generate an experiment execution and evaluation report."""
        cls._validate_format(format)
        session = get_session()
        try:
            exp_data = ExperimentTracker.get(experiment_id)
            report_id = uuid.uuid4().hex[:12]
            file_name = f"exp_{experiment_id}_{report_id}.{format}"
            out_path = REPORTS_DIR / file_name

            if format == "json":
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(exp_data, f, indent=2, default=_default_serializer)

            elif format == "csv":
                evals = exp_data.get("evaluations", [])
                if evals:
                    df_evals = pd.DataFrame(evals)
                else:
                    df_evals = pd.DataFrame([{
                        "experiment_id": exp_data["id"],
                        "algorithm": exp_data["algorithm"],
                        "status": exp_data["status"],
                    }])
                df_evals.to_csv(out_path, index=False)

            elif format == "pdf":
                cls._build_experiment_pdf(out_path, exp_data)

            file_size = out_path.stat().st_size
            report_rec = Report(
                id=report_id,
                dataset_id=exp_data.get("dataset_id"),
                experiment_id=experiment_id,
                report_type="experiment",
                format=format,
                file_path=str(out_path),
                file_size=file_size,
            )
            session.add(report_rec)
            session.commit()

            return cls._to_dict(report_rec)
        finally:
            session.close()

    @classmethod
    def generate_full_report(
        cls,
        dataset_id: str,
        format: str = "json",
    ) -> dict[str, Any]:
        """Generate a complete comprehensive report covering dataset, validation, EDA, and experiments."""
        cls._validate_format(format)
        session = get_session()
        try:
            ds = session.query(Dataset).filter_by(id=dataset_id).first()
            if not ds:
                raise ValueError(f"Dataset '{dataset_id}' not found.")

            val_rep = (
                session.query(ValidationReport)
                .filter_by(dataset_id=dataset_id)
                .order_by(ValidationReport.created_at.desc())
                .first()
            )
            eda_records = session.query(EDAResult).filter_by(dataset_id=dataset_id).all()
            experiments = ExperimentTracker.list_all(dataset_id=dataset_id)

            full_data = {
                "dataset": {
                    "id": ds.id,
                    "name": ds.name,
                    "num_rows": ds.num_rows,
                    "num_columns": ds.num_columns,
                    "validation_status": ds.validation_status,
                    "upload_time": ds.upload_time.isoformat() if ds.upload_time else None,
                },
                "validation": {
                    "status": val_rep.status if val_rep else "none",
                    "passed": val_rep.passed_checks if val_rep else 0,
                    "failed": val_rep.failed_checks if val_rep else 0,
                    "summary": val_rep.summary if val_rep else "",
                },
                "eda_modules": [r.analysis_type for r in eda_records],
                "experiments_count": len(experiments),
                "experiments": experiments,
            }

            report_id = uuid.uuid4().hex[:12]
            file_name = f"full_{dataset_id}_{report_id}.{format}"
            out_path = REPORTS_DIR / file_name

            if format == "json":
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(full_data, f, indent=2, default=_default_serializer)

            elif format == "csv":
                rows = [
                    {"section": "Dataset", "key": "Name", "value": ds.name},
                    {"section": "Dataset", "key": "Rows", "value": ds.num_rows},
                    {"section": "Dataset", "key": "Columns", "value": ds.num_columns},
                    {"section": "Validation", "key": "Status", "value": ds.validation_status},
                    {"section": "Experiments", "key": "Count", "value": len(experiments)},
                ]
                pd.DataFrame(rows).to_csv(out_path, index=False)

            elif format == "pdf":
                cls._build_full_pdf(out_path, full_data)

            file_size = out_path.stat().st_size
            report_rec = Report(
                id=report_id,
                dataset_id=dataset_id,
                report_type="full",
                format=format,
                file_path=str(out_path),
                file_size=file_size,
            )
            session.add(report_rec)
            session.commit()

            return cls._to_dict(report_rec)
        finally:
            session.close()

    @classmethod
    def get(cls, report_id: str) -> dict[str, Any]:
        """Retrieve report record by ID."""
        session = get_session()
        try:
            r = session.query(Report).filter_by(id=report_id).first()
            if not r:
                raise ReportNotFoundError(f"Report '{report_id}' does not exist.")
            return cls._to_dict(r)
        finally:
            session.close()

    @classmethod
    def list_all(
        cls,
        dataset_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        report_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """List reports matching optional filters."""
        session = get_session()
        try:
            query = session.query(Report)
            if dataset_id:
                query = query.filter_by(dataset_id=dataset_id)
            if experiment_id:
                query = query.filter_by(experiment_id=experiment_id)
            if report_type:
                query = query.filter_by(report_type=report_type)

            query = query.order_by(Report.created_at.desc())
            return [cls._to_dict(r) for r in query.all()]
        finally:
            session.close()

    @classmethod
    def delete(cls, report_id: str) -> bool:
        """Delete report record and remove file from disk."""
        session = get_session()
        try:
            r = session.query(Report).filter_by(id=report_id).first()
            if not r:
                raise ReportNotFoundError(f"Report '{report_id}' does not exist.")

            file_path = Path(r.file_path)
            if file_path.exists():
                file_path.unlink()

            session.delete(r)
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _validate_format(format: str):
        if format not in ("json", "csv", "pdf"):
            raise ValueError(f"Invalid format '{format}'. Supported formats: 'json', 'csv', 'pdf'.")

    # ── PDF Generation Helpers ───────────────────────────────────────────────

    @classmethod
    def _build_validation_pdf(cls, path: Path, data: dict[str, Any]):
        if not REPORTLAB_AVAILABLE:
            path.write_text(f"Validation Report: {json.dumps(data, indent=2)}")
            return

        doc = SimpleDocTemplate(str(path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = [
            Paragraph("<b>Universal AI/ML Studio — Dataset Validation Report</b>", styles["Heading1"]),
            Spacer(1, 12),
            Paragraph(f"<b>Dataset:</b> {data.get('dataset_name', 'Unknown')}", styles["Normal"]),
            Paragraph(f"<b>Status:</b> {data.get('status', 'Unknown').upper()}", styles["Normal"]),
            Paragraph(f"<b>Summary:</b> {data.get('summary', '')}", styles["Normal"]),
            Spacer(1, 14),
            Paragraph("<b>Check Results:</b>", styles["Heading3"]),
        ]

        table_data = [["Tier", "Check Name", "Status", "Message"]]
        for c in data.get("checks", [])[:35]:
            table_data.append([
                str(c.get("tier", "")),
                str(c.get("check_name", "")),
                str(c.get("status", "")),
                str(c.get("message", ""))[:45],
            ])

        if len(table_data) > 1:
            t = Table(table_data, colWidths=[60, 160, 70, 200])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B4C7E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ]))
            story.append(t)

        doc.build(story)

    @classmethod
    def _build_eda_pdf(cls, path: Path, data: dict[str, Any]):
        if not REPORTLAB_AVAILABLE:
            path.write_text(f"EDA Report: {json.dumps(data, indent=2)}")
            return

        doc = SimpleDocTemplate(str(path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = [
            Paragraph("<b>Universal AI/ML Studio — Exploratory Data Analysis Report</b>", styles["Heading1"]),
            Spacer(1, 12),
            Paragraph(f"<b>Dataset:</b> {data.get('dataset_name', 'Unknown')} ({data.get('rows', 0)} rows, {data.get('columns', 0)} columns)", styles["Normal"]),
            Spacer(1, 14),
            Paragraph("<b>Summary Statistics (Numerical):</b>", styles["Heading3"]),
        ]

        stats = data.get("eda", {}).get("statistics", {}).get("numerical", {})
        table_data = [["Column", "Mean", "Std", "Min", "50% (Median)", "Max"]]
        for col, s in list(stats.items())[:25]:
            table_data.append([
                col[:18],
                f"{s.get('mean', 0.0):.2f}",
                f"{s.get('std', 0.0):.2f}",
                f"{s.get('min', 0.0):.2f}",
                f"{s.get('50%', 0.0):.2f}",
                f"{s.get('max', 0.0):.2f}",
            ])

        if len(table_data) > 1:
            t = Table(table_data, colWidths=[120, 70, 70, 70, 80, 70])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B4C7E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ]))
            story.append(t)

        doc.build(story)

    @classmethod
    def _build_experiment_pdf(cls, path: Path, data: dict[str, Any]):
        if not REPORTLAB_AVAILABLE:
            path.write_text(f"Experiment Report: {json.dumps(data, indent=2)}")
            return

        doc = SimpleDocTemplate(str(path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = [
            Paragraph("<b>Universal AI/ML Studio — Experiment Report</b>", styles["Heading1"]),
            Spacer(1, 12),
            Paragraph(f"<b>Experiment:</b> {data.get('name', 'N/A')}", styles["Normal"]),
            Paragraph(f"<b>Algorithm:</b> {data.get('algorithm', '')} ({data.get('task_type', '')})", styles["Normal"]),
            Paragraph(f"<b>Status:</b> {data.get('status', '').upper()}", styles["Normal"]),
            Paragraph(f"<b>Training Time:</b> {data.get('training_time_sec', 0.0)} sec", styles["Normal"]),
            Spacer(1, 14),
            Paragraph("<b>Evaluation Metrics:</b>", styles["Heading3"]),
        ]

        evals = data.get("evaluations", [])
        table_data = [["Metric", "Value", "Split"]]
        for ev in evals:
            table_data.append([
                str(ev.get("metric_name", "")),
                f"{float(ev.get('metric_value', 0.0)):.4f}",
                str(ev.get("split", "")),
            ])

        if len(table_data) > 1:
            t = Table(table_data, colWidths=[180, 120, 100])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B4C7E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ]))
            story.append(t)

        doc.build(story)

    @classmethod
    def _build_full_pdf(cls, path: Path, data: dict[str, Any]):
        if not REPORTLAB_AVAILABLE:
            path.write_text(f"Full Report: {json.dumps(data, indent=2)}")
            return

        doc = SimpleDocTemplate(str(path), pagesize=letter)
        styles = getSampleStyleSheet()
        ds = data.get("dataset", {})
        val = data.get("validation", {})
        story = [
            Paragraph("<b>Universal AI/ML Studio — Comprehensive Study Report</b>", styles["Heading1"]),
            Spacer(1, 12),
            Paragraph(f"<b>Dataset:</b> {ds.get('name', 'N/A')} ({ds.get('num_rows', 0)} rows, {ds.get('num_columns', 0)} columns)", styles["Normal"]),
            Paragraph(f"<b>Validation Status:</b> {val.get('status', 'N/A').upper()} (Passed: {val.get('passed', 0)}, Failed: {val.get('failed', 0)})", styles["Normal"]),
            Paragraph(f"<b>Total Experiments Run:</b> {data.get('experiments_count', 0)}", styles["Normal"]),
            Spacer(1, 14),
        ]
        doc.build(story)

    @staticmethod
    def _to_dict(report: Report) -> dict[str, Any]:
        return {
            "id": report.id,
            "dataset_id": report.dataset_id,
            "experiment_id": report.experiment_id,
            "report_type": report.report_type,
            "format": report.format,
            "file_path": report.file_path,
            "file_size": report.file_size,
            "created_at": report.created_at.isoformat() if report.created_at else None,
        }

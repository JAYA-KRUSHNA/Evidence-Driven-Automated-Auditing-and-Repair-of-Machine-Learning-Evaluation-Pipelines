"""
EDA Module — Automated Multi-Format Report Generation Engine.

Exports full exploratory data analysis reports in:
1. HTML  – Modern, self-contained, responsive glassmorphic dashboard report
2. PDF   – Professional multi-page publication report using ReportLab
3. Excel – Formatted multi-tab workbook (.xlsx) using openpyxl
"""

from io import BytesIO
from typing import Any
import datetime
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


# ═══════════════════════════════════════════════════════════════════════════
#  1. HTML Report Generator
# ═══════════════════════════════════════════════════════════════════════════

def generate_html_report(dataset_name: str, eda_data: dict[str, Any]) -> str:
    """Generate a self-contained, responsive HTML report."""
    overview = eda_data.get("overview", {})
    quality = eda_data.get("quality", {})
    stats_data = eda_data.get("statistics", {})
    corr_data = eda_data.get("correlation", {})
    outlier_data = eda_data.get("outliers", eda_data.get("outlier_summary", {}))
    target_data = eda_data.get("target_analysis", {})
    features_data = eda_data.get("features", {})
    insights = eda_data.get("insights", [])

    health_score = overview.get("health_score", 100)
    health_grade = overview.get("health_grade", "Good")
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Generate insights rows
    insights_html = ""
    for ins in insights:
        sev = ins.get("severity", "info")
        badge_cls = "badge-danger" if sev == "critical" else "badge-warning" if sev == "warning" else "badge-success" if sev == "success" else "badge-info"
        insights_html += f"""
        <div class="insight-card">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <strong style="color:#f8fafc; font-size:15px;">{ins.get('finding')}</strong>
            <span class="badge {badge_cls}">{sev.upper()}</span>
          </div>
          <div style="color:#94a3b8; font-size:13px; margin-bottom:6px;"><strong>Evidence:</strong> {ins.get('evidence')}</div>
          <div style="color:#cbd5e1; font-size:13px; margin-bottom:6px;"><strong>Why it matters:</strong> {ins.get('explanation')}</div>
          <div style="color:#38bdf8; font-size:13px;"><strong>Recommendation:</strong> {ins.get('recommendation')}</div>
        </div>
        """

    # Generate stats rows
    num_stats = stats_data.get("numerical", {})
    stats_rows = ""
    for col, s in num_stats.items():
        stats_rows += f"""
        <tr>
          <td><strong>{col}</strong></td>
          <td>{s.get('count', 0):,}</td>
          <td>{s.get('mean', 0.0) if s.get('mean') is not None else '-'}</td>
          <td>{s.get('std', 0.0) if s.get('std') is not None else '-'}</td>
          <td>{s.get('min', 0.0) if s.get('min') is not None else '-'}</td>
          <td>{s.get('p25', s.get('25%', 0.0)) if s.get('p25') is not None else '-'}</td>
          <td>{s.get('median', 0.0) if s.get('median') is not None else '-'}</td>
          <td>{s.get('p75', s.get('75%', 0.0)) if s.get('p75') is not None else '-'}</td>
          <td>{s.get('max', 0.0) if s.get('max') is not None else '-'}</td>
          <td>{s.get('skewness', 0.0) if s.get('skewness') is not None else '-'}</td>
        </tr>
        """

    # Quality issues
    issues = quality.get("issues", [])
    quality_rows = ""
    for iss in issues[:15]:
        s_cls = "badge-danger" if iss.get("severity") == "critical" else "badge-warning" if iss.get("severity") == "warning" else "badge-info"
        quality_rows += f"""
        <tr>
          <td><span class="badge {s_cls}">{iss.get('severity', 'info').upper()}</span></td>
          <td>{iss.get('column') or 'Dataset-wide'}</td>
          <td>{iss.get('type', '')}</td>
          <td>{iss.get('message', '')}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Universal AI/ML Studio — EDA Report: {dataset_name}</title>
  <style>
    :root {{
      --bg: #090d16;
      --card-bg: rgba(30, 41, 59, 0.7);
      --border: rgba(255, 255, 255, 0.08);
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --success: #34d399;
      --warning: #fbbf24;
      --danger: #f43f5e;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 30px;
      line-height: 1.5;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    .header {{
      background: linear-gradient(135deg, rgba(56, 189, 248, 0.1), rgba(129, 140, 248, 0.1));
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .title {{ font-size: 24px; font-weight: 700; color: #fff; margin: 0 0 4px 0; }}
    .subtitle {{ color: var(--text-muted); font-size: 14px; margin: 0; }}
    .health-badge {{
      background: rgba(52, 211, 153, 0.15);
      border: 1px solid #34d399;
      color: #34d399;
      padding: 10px 18px;
      border-radius: 10px;
      text-align: center;
    }}
    .health-num {{ font-size: 28px; font-weight: 800; }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .kpi-box {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
    }}
    .kpi-label {{ color: var(--text-muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
    .kpi-val {{ font-size: 22px; font-weight: 700; color: #fff; margin-top: 4px; }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 24px;
    }}
    .card-title {{ font-size: 17px; font-weight: 600; margin: 0 0 16px 0; color: #fff; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); }}
    th {{ background: rgba(255, 255, 255, 0.04); color: var(--text-muted); font-weight: 600; }}
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 700;
    }}
    .badge-success {{ background: rgba(52, 211, 153, 0.2); color: #34d399; }}
    .badge-warning {{ background: rgba(251, 191, 36, 0.2); color: #fbbf24; }}
    .badge-danger {{ background: rgba(244, 63, 94, 0.2); color: #f43f5e; }}
    .badge-info {{ background: rgba(56, 189, 248, 0.2); color: #38bdf8; }}
    .insight-card {{
      background: rgba(255, 255, 255, 0.02);
      border-left: 4px solid #38bdf8;
      border-radius: 4px 8px 8px 4px;
      padding: 14px;
      margin-bottom: 12px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <h1 class="title">📊 Automated EDA Audit Report</h1>
        <p class="subtitle">Dataset: <strong>{dataset_name}</strong> | Generated: {now_str} | Universal AI/ML Studio</p>
      </div>
      <div class="health-badge">
        <div class="health-num">{health_score:.1f}</div>
        <div style="font-size:12px; font-weight:600;">{health_grade.upper()} HEALTH</div>
      </div>
    </div>

    <!-- KPIs -->
    <div class="kpi-grid">
      <div class="kpi-box">
        <div class="kpi-label">Dimensions</div>
        <div class="kpi-val">{overview.get('num_rows', 0):,} × {overview.get('num_columns', 0):,}</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-label">Missing Data</div>
        <div class="kpi-val">{overview.get('missing_pct', 0.0):.2f}%</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-label">Duplicates</div>
        <div class="kpi-val">{overview.get('duplicate_rows', 0):,} rows</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-label">Memory Usage</div>
        <div class="kpi-val">{overview.get('memory_mb', 0.0):.2f} MB</div>
      </div>
    </div>

    <!-- AI Insights -->
    <div class="card">
      <h2 class="card-title">💡 Intelligent AI Insights & Recommendations</h2>
      {insights_html if insights_html else '<p style="color:var(--text-muted)">No critical warnings or defects identified.</p>'}
    </div>

    <!-- Data Quality Audit -->
    <div class="card">
      <h2 class="card-title">🛡️ Data Quality Audit Log</h2>
      <div style="overflow-x:auto;">
        <table>
          <thead><tr><th>Severity</th><th>Column</th><th>Issue Type</th><th>Description</th></tr></thead>
          <tbody>{quality_rows if quality_rows else '<tr><td colspan="4" style="text-align:center; color:var(--text-muted)">No quality issues detected.</td></tr>'}</tbody>
        </table>
      </div>
    </div>

    <!-- Numerical Stats -->
    <div class="card">
      <h2 class="card-title">📈 Numerical Descriptive Statistics</h2>
      <div style="overflow-x:auto;">
        <table>
          <thead><tr><th>Column</th><th>Count</th><th>Mean</th><th>Std</th><th>Min</th><th>Q1</th><th>Median</th><th>Q3</th><th>Max</th><th>Skew</th></tr></thead>
          <tbody>{stats_rows if stats_rows else '<tr><td colspan="10" style="text-align:center; color:var(--text-muted)">No numerical features present.</td></tr>'}</tbody>
        </table>
      </div>
    </div>
  </div>
</body>
</html>"""
    return html


# ═══════════════════════════════════════════════════════════════════════════
#  2. PDF Report Generator (ReportLab)
# ═══════════════════════════════════════════════════════════════════════════

def generate_pdf_report(dataset_name: str, eda_data: dict[str, Any]) -> bytes:
    """Generate a clean, professional multi-page PDF document."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=20,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=15,
    )
    h2_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=14,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )
    bold_body = ParagraphStyle(
        "BoldBody",
        parent=body_style,
        fontName="Helvetica-Bold",
    )

    story = []
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    overview = eda_data.get("overview", {})
    health_score = overview.get("health_score", 100)
    health_grade = overview.get("health_grade", "Good")

    # Header
    story.append(Paragraph(f"Universal AI/ML Studio — Automated EDA Audit", title_style))
    story.append(Paragraph(f"Dataset: <b>{dataset_name}</b> | Generated: {now_str} | Health Score: <b>{health_score:.1f}/100 ({health_grade})</b>", subtitle_style))
    story.append(Spacer(1, 10))

    # Overview KPI Table
    kpi_data = [
        ["Rows", f"{overview.get('num_rows', 0):,}", "Columns", f"{overview.get('num_columns', 0):,}"],
        ["Missing Cells", f"{overview.get('total_missing_cells', 0):,} ({overview.get('missing_pct', 0):.1f}%)", "Duplicate Rows", f"{overview.get('duplicate_rows', 0):,}"],
        ["Numeric Features", str(overview.get("num_numerical", len(overview.get("numerical_columns", [])))), "Categorical Features", str(overview.get("num_categorical", len(overview.get("categorical_columns", []))))],
        ["Memory Footprint", f"{overview.get('memory_mb', 0):.2f} MB", "Health Grade", f"{health_grade} ({health_score:.1f}/100)"],
    ]
    t_kpi = Table(kpi_data, colWidths=[110, 150, 110, 170])
    t_kpi.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1e293b")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_kpi)
    story.append(Spacer(1, 15))

    # AI Insights
    story.append(Paragraph("Intelligent AI Findings & Recommendations", h2_style))
    insights = eda_data.get("insights", [])[:8]
    if insights:
        ins_data = [["Sev", "Finding", "Evidence & Action"]]
        for ins in insights:
            ins_data.append([
                ins.get("severity", "info").upper()[:4],
                Paragraph(f"<b>{ins.get('finding')}</b>", body_style),
                Paragraph(f"{ins.get('evidence')}<br/><b>Action:</b> {ins.get('recommendation')}", body_style),
            ])
        t_ins = Table(ins_data, colWidths=[40, 220, 280])
        t_ins.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
            ("PADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(t_ins)
    else:
        story.append(Paragraph("No critical issues or outliers detected.", body_style))

    # Page Break for Statistics
    story.append(PageBreak())

    # Numerical Summary Table
    story.append(Paragraph("Numerical Descriptive Statistics", h2_style))
    num_stats = eda_data.get("statistics", {}).get("numerical", {})
    if num_stats:
        stats_table_data = [["Feature", "Count", "Mean", "Std", "Min", "Median", "Max", "Skew"]]
        for col, s in list(num_stats.items())[:20]:
            stats_table_data.append([
                str(col)[:18],
                f"{s.get('count', 0):,}",
                f"{s.get('mean', 0.0):.2f}" if s.get("mean") is not None else "-",
                f"{s.get('std', 0.0):.2f}" if s.get("std") is not None else "-",
                f"{s.get('min', 0.0):.2f}" if s.get("min") is not None else "-",
                f"{s.get('median', 0.0):.2f}" if s.get("median") is not None else "-",
                f"{s.get('max', 0.0):.2f}" if s.get("max") is not None else "-",
                f"{s.get('skewness', 0.0):.2f}" if s.get("skewness") is not None else "-",
            ])
        t_stats = Table(stats_table_data, colWidths=[100, 55, 60, 60, 60, 65, 70, 70])
        t_stats.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
            ("PADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t_stats)
    else:
        story.append(Paragraph("No numerical columns present.", body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ═══════════════════════════════════════════════════════════════════════════
#  3. Excel Report Generator (openpyxl)
# ═══════════════════════════════════════════════════════════════════════════

def generate_excel_report(dataset_name: str, eda_data: dict[str, Any]) -> bytes:
    """Generate a multi-tab formatted Excel workbook."""
    wb = openpyxl.Workbook()
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
    thin_border = Border(
        left=Side(style="thin", color="CBD5E1"),
        right=Side(style="thin", color="CBD5E1"),
        top=Side(style="thin", color="CBD5E1"),
        bottom=Side(style="thin", color="CBD5E1"),
    )

    overview = eda_data.get("overview", {})

    # Sheet 1: Overview
    ws_ov = wb.active
    ws_ov.title = "Overview & Health"
    ws_ov.append(["Universal AI/ML Studio — Automated EDA Audit"])
    ws_ov.append(["Dataset", dataset_name])
    ws_ov.append(["Generated", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    ws_ov.append(["Health Score (0-100)", overview.get("health_score", 100)])
    ws_ov.append(["Health Grade", overview.get("health_grade", "Good")])
    ws_ov.append([])
    ws_ov.append(["Metric", "Value"])
    ws_ov.append(["Total Rows", overview.get("num_rows", 0)])
    ws_ov.append(["Total Columns", overview.get("num_columns", 0)])
    ws_ov.append(["Total Cells", overview.get("total_cells", 0)])
    ws_ov.append(["Missing Cells", overview.get("total_missing_cells", 0)])
    ws_ov.append(["Missing Percentage", f"{overview.get('missing_pct', 0.0):.2f}%"])
    ws_ov.append(["Duplicate Rows", overview.get("duplicate_rows", 0)])
    ws_ov.append(["Memory Consumption (MB)", overview.get("memory_mb", 0.0)])

    # Sheet 2: Descriptive Statistics
    ws_stat = wb.create_sheet(title="Descriptive Statistics")
    ws_stat.append(["Column", "Count", "Mean", "Std", "Min", "25%", "Median", "75%", "Max", "Skewness", "Kurtosis", "Normality"])
    for cell in ws_stat[1]:
        cell.font = header_font
        cell.fill = header_fill

    num_stats = eda_data.get("statistics", {}).get("numerical", {})
    for col, s in num_stats.items():
        ws_stat.append([
            col,
            s.get("count", 0),
            s.get("mean"),
            s.get("std"),
            s.get("min"),
            s.get("p25", s.get("25%")),
            s.get("median"),
            s.get("p75", s.get("75%")),
            s.get("max"),
            s.get("skewness"),
            s.get("kurtosis"),
            "Normal" if s.get("is_normal") else "Non-normal",
        ])

    # Sheet 3: Quality Audit
    ws_qual = wb.create_sheet(title="Quality Audit")
    ws_qual.append(["Severity", "Column", "Issue Type", "Description"])
    for cell in ws_qual[1]:
        cell.font = header_font
        cell.fill = header_fill

    issues = eda_data.get("quality", {}).get("issues", [])
    for iss in issues:
        ws_qual.append([
            iss.get("severity", "info").upper(),
            iss.get("column") or "Dataset-wide",
            iss.get("type", ""),
            iss.get("message", ""),
        ])

    # Sheet 4: Outliers Summary
    ws_out = wb.create_sheet(title="Outlier Detection")
    ws_out.append(["Column", "IQR Outliers", "IQR %", "Lower Bound", "Upper Bound", "Z-Score Outliers", "Z-Score %"])
    for cell in ws_out[1]:
        cell.font = header_font
        cell.fill = header_fill

    out_cols = eda_data.get("outliers", eda_data.get("outlier_summary", {})).get("per_column", {})
    for col, o in out_cols.items():
        ws_out.append([
            col,
            o.get("iqr_outliers", 0),
            o.get("iqr_outlier_pct", 0.0),
            o.get("iqr_lower_bound"),
            o.get("iqr_upper_bound"),
            o.get("zscore_outliers", 0),
            o.get("zscore_outlier_pct", 0.0),
        ])

    # Sheet 5: AI Insights
    ws_ins = wb.create_sheet(title="AI Insights")
    ws_ins.append(["Category", "Severity", "Finding", "Evidence", "Explanation", "Action Recommendation"])
    for cell in ws_ins[1]:
        cell.font = header_font
        cell.fill = header_fill

    insights = eda_data.get("insights", [])
    for ins in insights:
        ws_ins.append([
            ins.get("category", "General").title(),
            ins.get("severity", "info").upper(),
            ins.get("finding", ""),
            ins.get("evidence", ""),
            ins.get("explanation", ""),
            ins.get("recommendation", ""),
        ])

    # Auto-adjust column widths
    for sheet in wb.worksheets:
        for col in sheet.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            col_letter = openpyxl.utils.get_column_letter(col[0].column)
            sheet.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 50)

    buffer = BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()

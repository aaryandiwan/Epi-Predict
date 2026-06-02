"""
Epi Predict – Report Generator

Produces structured weekly influenza surveillance reports in three
output formats:

    1. **Dict** – machine-readable; consumed by the REST API.
    2. **Plain text** – terminal / log-friendly; attached to alert emails.
    3. **HTML** – rendered inside the Streamlit dashboard.

Report Sections:
    ┌────────────────────────────┐
    │ Executive Summary          │
    │ Risk Assessment            │
    │ Forecast Summary           │
    │ Recommendations            │
    │ Data Quality Notes         │
    └────────────────────────────┘

Author : Epi Predict Team
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from config.settings import RISK_THRESHOLDS

logger = logging.getLogger("epi_predict.report_generator")


# ─── Report Generation ──────────────────────────────────────────────────────

def generate_weekly_report(
    predictions: List[float],
    risk_level: str,
    recommendations: Union[Dict[str, Any], List[str]],
    date: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a complete weekly influenza surveillance report.

    The report aggregates predictions, risk assessment, and
    recommendations into five structured sections suitable for
    stakeholder distribution.

    Args:
        predictions: List of predicted case counts for the reporting
            period (typically 1-12 weeks of forecast data).
        risk_level: Current risk classification key
            (``"low"`` | ``"moderate"`` | ``"high"`` | ``"severe"``).
        recommendations: Either a dict with an ``"actions"`` key (as
            returned by :func:`modules.recommendation_engine.get_recommendations`)
            or a flat list of recommendation strings.
        date: ISO-format date string for the report header. Defaults
            to the current UTC date.

    Returns:
        Report dictionary with keys:
            - **report_id** (str): Unique report identifier.
            - **generated_at** (str): ISO timestamp.
            - **report_date** (str): The reporting date.
            - **sections** (dict): Five report sections (see below).
            - **metadata** (dict): Summary statistics.

    Sections detail:
        - **executive_summary**: High-level overview paragraph.
        - **risk_assessment**: Risk level, colour, icon, threshold info.
        - **forecast_summary**: Statistics and week-by-week forecast.
        - **recommendations**: Action items for the current risk tier.
        - **data_quality**: Notes on data completeness and caveats.
    """
    now = datetime.now(timezone.utc)
    report_date = date or now.strftime("%Y-%m-%d")
    report_id = f"RPT-{now.strftime('%Y%m%d-%H%M%S')}"

    level_key = risk_level.strip().lower()
    threshold = RISK_THRESHOLDS.get(level_key, {})

    # ── Statistics ───────────────────────────────────────────────────────
    if predictions:
        max_pred = max(predictions)
        min_pred = min(predictions)
        avg_pred = sum(predictions) / len(predictions)
    else:
        max_pred = min_pred = avg_pred = 0.0

    # ── Trend direction ──────────────────────────────────────────────────
    trend = _compute_trend(predictions)

    # ── Normalise recommendations ────────────────────────────────────────
    if isinstance(recommendations, dict):
        rec_actions: List[str] = recommendations.get("actions", [])
        rec_summary: str = recommendations.get("summary", "")
        rec_urgency: str = recommendations.get("urgency", "low")
    else:
        rec_actions = list(recommendations) if recommendations else []
        rec_summary = ""
        rec_urgency = level_key

    # ── Executive Summary ────────────────────────────────────────────────
    executive_summary = (
        f"Weekly Influenza Surveillance Report generated on {report_date}. "
        f"The current risk level is assessed as "
        f"{threshold.get('label', level_key.title())} "
        f"({threshold.get('icon', '')}).  "
        f"Over the forecast window of {len(predictions)} week(s), the "
        f"predicted case count ranges from {min_pred:,.0f} to "
        f"{max_pred:,.0f} (average {avg_pred:,.0f}).  "
        f"The overall trend is {trend}.  "
        f"{'Immediate action is recommended.' if level_key in ('high', 'severe') else 'Standard precautions apply.'}"
    )

    # ── Build report ─────────────────────────────────────────────────────
    report: Dict[str, Any] = {
        "report_id": report_id,
        "generated_at": now.isoformat(),
        "report_date": report_date,
        "sections": {
            "executive_summary": {
                "title": "Executive Summary",
                "content": executive_summary,
            },
            "risk_assessment": {
                "title": "Risk Assessment",
                "risk_level": level_key,
                "label": threshold.get("label", level_key.title()),
                "color": threshold.get("color", "#6b7280"),
                "icon": threshold.get("icon", "ℹ️"),
                "threshold_range": (
                    f"{threshold.get('min', 0):,} – "
                    f"{'∞' if threshold.get('max') == float('inf') else f'{threshold.get(\"max\", 0):,}'}"
                ),
                "description": (
                    f"The current influenza activity corresponds to "
                    f"{threshold.get('label', 'Unknown')} status based on "
                    f"predicted case counts within the "
                    f"{threshold.get('min', 0):,} – "
                    f"{'∞' if threshold.get('max') == float('inf') else f'{threshold.get(\"max\", 0):,}'} range."
                ),
            },
            "forecast_summary": {
                "title": "Forecast Summary",
                "num_weeks": len(predictions),
                "max_predicted": round(max_pred, 2),
                "min_predicted": round(min_pred, 2),
                "avg_predicted": round(avg_pred, 2),
                "trend": trend,
                "weekly_predictions": [
                    {"week": i + 1, "predicted_cases": round(p, 2)}
                    for i, p in enumerate(predictions)
                ],
            },
            "recommendations": {
                "title": "Recommendations",
                "urgency": rec_urgency,
                "summary": rec_summary,
                "actions": rec_actions,
            },
            "data_quality": {
                "title": "Data Quality Notes",
                "notes": [
                    "Predictions are based on historical WHO FluNet surveillance data.",
                    "Model accuracy depends on the completeness and timeliness of reported data.",
                    "Actual case counts may differ from predictions due to reporting delays.",
                    f"This report covers a forecast window of {len(predictions)} week(s).",
                    "Confidence intervals are not included in this summary; "
                    "refer to the detailed forecast endpoint for interval data.",
                ],
            },
        },
        "metadata": {
            "total_predictions": len(predictions),
            "risk_level": level_key,
            "trend": trend,
            "max_predicted": round(max_pred, 2),
            "avg_predicted": round(avg_pred, 2),
        },
    }

    logger.info(
        "Report %s generated: date=%s, risk=%s, predictions=%d, trend=%s",
        report_id,
        report_date,
        level_key,
        len(predictions),
        trend,
    )

    return report


# ─── Plain-Text Formatter ───────────────────────────────────────────────────

def format_report_text(report: Dict[str, Any]) -> str:
    """Render a report dictionary as formatted plain text.

    Suitable for terminal output, log files, and plain-text email
    attachments.

    Args:
        report: Report dict as returned by :func:`generate_weekly_report`.

    Returns:
        Multi-line formatted text string.
    """
    sections = report.get("sections", {})
    width = 70
    sep = "=" * width
    thin = "-" * width

    lines: List[str] = [
        sep,
        "  EPI PREDICT – WEEKLY INFLUENZA SURVEILLANCE REPORT".center(width),
        sep,
        f"  Report ID   : {report.get('report_id', 'N/A')}",
        f"  Report Date : {report.get('report_date', 'N/A')}",
        f"  Generated   : {report.get('generated_at', 'N/A')}",
        sep,
        "",
    ]

    # Executive Summary
    exec_sec = sections.get("executive_summary", {})
    lines += [
        f"  {exec_sec.get('title', 'Executive Summary').upper()}",
        thin,
        _wrap_text(exec_sec.get("content", ""), indent=4, width=width),
        "",
    ]

    # Risk Assessment
    risk_sec = sections.get("risk_assessment", {})
    lines += [
        f"  {risk_sec.get('title', 'Risk Assessment').upper()}",
        thin,
        f"    Risk Level   : {risk_sec.get('icon', '')} {risk_sec.get('label', 'N/A')}",
        f"    Threshold    : {risk_sec.get('threshold_range', 'N/A')} cases",
        f"    Description  : {risk_sec.get('description', '')}",
        "",
    ]

    # Forecast Summary
    fc_sec = sections.get("forecast_summary", {})
    lines += [
        f"  {fc_sec.get('title', 'Forecast Summary').upper()}",
        thin,
        f"    Forecast Window : {fc_sec.get('num_weeks', 0)} weeks",
        f"    Max Predicted   : {fc_sec.get('max_predicted', 0):,.2f}",
        f"    Min Predicted   : {fc_sec.get('min_predicted', 0):,.2f}",
        f"    Avg Predicted   : {fc_sec.get('avg_predicted', 0):,.2f}",
        f"    Trend           : {fc_sec.get('trend', 'N/A')}",
        "",
    ]

    # Weekly breakdown
    weekly = fc_sec.get("weekly_predictions", [])
    if weekly:
        lines.append("    Week-by-Week Forecast:")
        for w in weekly:
            lines.append(
                f"      Week {w['week']:>2d} : {w['predicted_cases']:>10,.2f} cases"
            )
        lines.append("")

    # Recommendations
    rec_sec = sections.get("recommendations", {})
    lines += [
        f"  {rec_sec.get('title', 'Recommendations').upper()}",
        thin,
        f"    Urgency: {rec_sec.get('urgency', 'N/A').upper()}",
    ]
    if rec_sec.get("summary"):
        lines.append(f"    {rec_sec['summary']}")
    for i, action in enumerate(rec_sec.get("actions", []), 1):
        lines.append(f"    {i}. {action}")
    lines.append("")

    # Data Quality
    dq_sec = sections.get("data_quality", {})
    lines += [
        f"  {dq_sec.get('title', 'Data Quality Notes').upper()}",
        thin,
    ]
    for note in dq_sec.get("notes", []):
        lines.append(f"    • {note}")
    lines += ["", sep]

    text = "\n".join(lines)
    logger.debug("Formatted text report (%d chars).", len(text))
    return text


# ─── HTML Formatter ──────────────────────────────────────────────────────────

def format_report_html(report: Dict[str, Any]) -> str:
    """Render a report dictionary as styled HTML.

    Designed for embedding in the Streamlit dashboard via
    ``st.markdown(..., unsafe_allow_html=True)``.

    Args:
        report: Report dict as returned by :func:`generate_weekly_report`.

    Returns:
        HTML string.
    """
    sections = report.get("sections", {})
    risk_sec = sections.get("risk_assessment", {})
    exec_sec = sections.get("executive_summary", {})
    fc_sec = sections.get("forecast_summary", {})
    rec_sec = sections.get("recommendations", {})
    dq_sec = sections.get("data_quality", {})

    color = risk_sec.get("color", "#6b7280")
    icon = risk_sec.get("icon", "ℹ️")

    # Weekly predictions table rows
    weekly_rows = ""
    for w in fc_sec.get("weekly_predictions", []):
        weekly_rows += (
            f"<tr><td style='padding:6px 12px;'>Week {w['week']}</td>"
            f"<td style='padding:6px 12px; text-align:right;'>"
            f"{w['predicted_cases']:,.2f}</td></tr>"
        )

    # Recommendation list items
    rec_items = "".join(
        f"<li style='margin-bottom:6px;'>{a}</li>"
        for a in rec_sec.get("actions", [])
    )

    # Data quality bullets
    dq_items = "".join(
        f"<li style='margin-bottom:4px; color:#6b7280;'>{n}</li>"
        for n in dq_sec.get("notes", [])
    )

    html = f"""
    <div style="font-family:'Segoe UI',Arial,sans-serif; max-width:800px; margin:auto;">
        <!-- Header -->
        <div style="background:linear-gradient(135deg,{color},#1e293b); color:white;
                    padding:24px; border-radius:12px 12px 0 0; text-align:center;">
            <h1 style="margin:0; font-size:24px;">
                {icon} Epi Predict – Weekly Report
            </h1>
            <p style="margin:8px 0 0; opacity:0.9;">
                Report Date: {report.get('report_date', 'N/A')} &nbsp;|&nbsp;
                ID: {report.get('report_id', 'N/A')}
            </p>
        </div>

        <div style="border:1px solid #e5e7eb; border-top:none;
                    border-radius:0 0 12px 12px; padding:24px;">

            <!-- Executive Summary -->
            <h2 style="color:#1e293b; border-bottom:2px solid {color};
                       padding-bottom:8px;">📋 Executive Summary</h2>
            <p style="line-height:1.7; color:#374151;">
                {exec_sec.get('content', '')}
            </p>

            <!-- Risk Assessment -->
            <h2 style="color:#1e293b; border-bottom:2px solid {color};
                       padding-bottom:8px;">🎯 Risk Assessment</h2>
            <div style="display:flex; gap:16px; flex-wrap:wrap; margin-bottom:16px;">
                <div style="flex:1; min-width:200px; padding:16px;
                            background:{color}22; border-left:4px solid {color};
                            border-radius:8px;">
                    <strong>Risk Level</strong><br>
                    <span style="font-size:20px;">{icon} {risk_sec.get('label', 'N/A')}</span>
                </div>
                <div style="flex:1; min-width:200px; padding:16px;
                            background:#f8fafc; border-left:4px solid #94a3b8;
                            border-radius:8px;">
                    <strong>Threshold Range</strong><br>
                    <span style="font-size:20px;">{risk_sec.get('threshold_range', 'N/A')} cases</span>
                </div>
            </div>
            <p style="color:#6b7280;">{risk_sec.get('description', '')}</p>

            <!-- Forecast Summary -->
            <h2 style="color:#1e293b; border-bottom:2px solid {color};
                       padding-bottom:8px;">📊 Forecast Summary</h2>
            <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:16px;">
                <div style="flex:1; min-width:140px; text-align:center; padding:12px;
                            background:#f0fdf4; border-radius:8px;">
                    <div style="font-size:12px; color:#6b7280;">Min</div>
                    <div style="font-size:22px; font-weight:bold; color:#16a34a;">
                        {fc_sec.get('min_predicted', 0):,.0f}
                    </div>
                </div>
                <div style="flex:1; min-width:140px; text-align:center; padding:12px;
                            background:#eff6ff; border-radius:8px;">
                    <div style="font-size:12px; color:#6b7280;">Average</div>
                    <div style="font-size:22px; font-weight:bold; color:#2563eb;">
                        {fc_sec.get('avg_predicted', 0):,.0f}
                    </div>
                </div>
                <div style="flex:1; min-width:140px; text-align:center; padding:12px;
                            background:#fef2f2; border-radius:8px;">
                    <div style="font-size:12px; color:#6b7280;">Max</div>
                    <div style="font-size:22px; font-weight:bold; color:#dc2626;">
                        {fc_sec.get('max_predicted', 0):,.0f}
                    </div>
                </div>
                <div style="flex:1; min-width:140px; text-align:center; padding:12px;
                            background:#faf5ff; border-radius:8px;">
                    <div style="font-size:12px; color:#6b7280;">Trend</div>
                    <div style="font-size:22px; font-weight:bold; color:#7c3aed;">
                        {'📈' if fc_sec.get('trend') == 'increasing' else '📉' if fc_sec.get('trend') == 'decreasing' else '➡️'}
                        {fc_sec.get('trend', 'N/A').title()}
                    </div>
                </div>
            </div>
            {'<table style="width:100%; border-collapse:collapse; margin-bottom:16px;">'
             '<tr style="background:#f1f5f9;"><th style="padding:8px 12px; text-align:left;">Week</th>'
             '<th style="padding:8px 12px; text-align:right;">Predicted Cases</th></tr>'
             + weekly_rows + '</table>' if weekly_rows else ''}

            <!-- Recommendations -->
            <h2 style="color:#1e293b; border-bottom:2px solid {color};
                       padding-bottom:8px;">💡 Recommendations</h2>
            <div style="padding:12px 16px; background:#fffbeb; border-left:4px solid #f59e0b;
                        border-radius:8px; margin-bottom:12px;">
                <strong>Urgency: {rec_sec.get('urgency', 'N/A').upper()}</strong>
                {f"<br>{rec_sec['summary']}" if rec_sec.get('summary') else ''}
            </div>
            <ol style="line-height:1.8;">{rec_items}</ol>

            <!-- Data Quality -->
            <h2 style="color:#1e293b; border-bottom:2px solid {color};
                       padding-bottom:8px;">📝 Data Quality Notes</h2>
            <ul style="line-height:1.7;">{dq_items}</ul>

            <!-- Footer -->
            <hr style="margin:24px 0; border:none; border-top:1px solid #e5e7eb;">
            <p style="font-size:12px; color:#9ca3af; text-align:center;">
                Generated by Epi Predict – Influenza Outbreak Early Warning System
                &nbsp;|&nbsp; {report.get('generated_at', '')}
            </p>
        </div>
    </div>
    """

    logger.debug("Formatted HTML report (%d chars).", len(html))
    return html


# ─── Private Helpers ─────────────────────────────────────────────────────────

def _compute_trend(predictions: List[float]) -> str:
    """Determine trend direction from a prediction list."""
    if not predictions or len(predictions) < 2:
        return "stable"

    mid = len(predictions) // 2
    first_half = sum(predictions[:mid]) / max(mid, 1)
    second_half = sum(predictions[mid:]) / max(len(predictions) - mid, 1)
    diff_pct = ((second_half - first_half) / max(first_half, 1.0)) * 100

    if diff_pct > 10:
        return "increasing"
    elif diff_pct < -10:
        return "decreasing"
    return "stable"


def _wrap_text(text: str, indent: int = 4, width: int = 70) -> str:
    """Simple word-wrapping for plain-text output."""
    import textwrap

    return textwrap.fill(
        text,
        width=width,
        initial_indent=" " * indent,
        subsequent_indent=" " * indent,
    )

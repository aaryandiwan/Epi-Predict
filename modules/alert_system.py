"""
Epi Predict – Alert & Notification System

Monitors risk-level transitions and fires alerts when the influenza
situation *escalates* (i.e. the risk level moves to a more severe tier).
Alerts are persisted to a JSON log file and can optionally be dispatched
as email notifications via SMTP.

Alert Severity Levels:
    ┌────────────┬────────────────────────────────────────────┐
    │ Level      │ Trigger                                    │
    ├────────────┼────────────────────────────────────────────┤
    │ INFO       │ Risk stayed the same or decreased          │
    │ WARNING    │ Risk escalated to *moderate*               │
    │ CRITICAL   │ Risk escalated to *high*                   │
    │ EMERGENCY  │ Risk escalated to *severe*                 │
    └────────────┴────────────────────────────────────────────┘

Author : Epi Predict Team
"""

from __future__ import annotations

import json
import logging
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import (
    LOGS_DIR,
    ALERT_EMAIL_ENABLED,
    SMTP_HOST,
    SMTP_PORT,
    SMTP_USER,
    SMTP_PASSWORD,
    ALERT_RECIPIENTS,
    RISK_THRESHOLDS,
)

logger = logging.getLogger("epi_predict.alert_system")

# ─── Constants ───────────────────────────────────────────────────────────────

# Ordered from lowest to highest severity
_SEVERITY_ORDER: List[str] = ["low", "moderate", "high", "severe"]

# Maps a *new* risk level to the alert severity that should fire
_ALERT_LEVEL_MAP: Dict[str, str] = {
    "low": "INFO",
    "moderate": "WARNING",
    "high": "CRITICAL",
    "severe": "EMERGENCY",
}

_ALERT_LOG_FILE: Path = LOGS_DIR / "alert_history.json"


# ─── Alert System Class ─────────────────────────────────────────────────────

class AlertSystem:
    """Outbreak alert manager with escalation detection and notification.

    Usage::

        alerts = AlertSystem()
        alert = alerts.check_alert(current_risk="high", previous_risk="moderate")
        if alert:
            alerts.send_notification(alert)
    """

    def __init__(self) -> None:
        """Initialise the alert system and ensure the log file exists."""
        self._ensure_log_file()
        logger.info(
            "AlertSystem initialised. Email enabled: %s, log: %s",
            ALERT_EMAIL_ENABLED,
            _ALERT_LOG_FILE,
        )

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _ensure_log_file() -> None:
        """Create the alert log file with an empty list if absent."""
        if not _ALERT_LOG_FILE.exists():
            _ALERT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            _ALERT_LOG_FILE.write_text("[]", encoding="utf-8")
            logger.debug("Created empty alert log at %s", _ALERT_LOG_FILE)

    @staticmethod
    def _severity_index(risk_level: str) -> int:
        """Return numeric index for a risk level (0 = low, 3 = severe)."""
        try:
            return _SEVERITY_ORDER.index(risk_level.strip().lower())
        except ValueError:
            logger.warning("Unknown risk level '%s'; treating as index -1.", risk_level)
            return -1

    def _save_alert(self, alert: Dict[str, Any]) -> None:
        """Append an alert to the persistent JSON log."""
        try:
            history = self.get_alert_history()
            history.append(alert)
            _ALERT_LOG_FILE.write_text(
                json.dumps(history, indent=2, default=str), encoding="utf-8"
            )
            logger.debug("Alert saved to log (total: %d).", len(history))
        except Exception as exc:
            logger.error("Failed to save alert: %s", exc, exc_info=True)

    # ── Public API ───────────────────────────────────────────────────────

    def check_alert(
        self,
        current_risk: str,
        previous_risk: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate whether the current risk level warrants an alert.

        An alert is generated when the risk level has *escalated*
        (moved to a higher tier) compared to the previous assessment.
        If no *previous_risk* is supplied, any level above ``"low"``
        triggers an alert.

        Args:
            current_risk: Current risk classification key
                (``"low"`` | ``"moderate"`` | ``"high"`` | ``"severe"``).
            previous_risk: Previous risk classification key (optional).

        Returns:
            Alert dictionary if escalation detected, otherwise ``None``.
            Alert schema::

                {
                    "alert_id": "ALT-20260602-153012-001",
                    "timestamp": "2026-06-02T15:30:12+00:00",
                    "alert_level": "CRITICAL",
                    "current_risk": "high",
                    "previous_risk": "moderate",
                    "escalated": True,
                    "message": "...",
                    "recommendations": "...",
                }
        """
        current = current_risk.strip().lower()
        previous = previous_risk.strip().lower() if previous_risk else None

        current_idx = self._severity_index(current)
        previous_idx = self._severity_index(previous) if previous else -1

        escalated = current_idx > previous_idx
        timestamp = datetime.now(timezone.utc)

        # Only generate alert on escalation (or first assessment above low)
        if not escalated:
            logger.info(
                "No escalation: current=%s (%d), previous=%s (%d).",
                current, current_idx,
                previous or "N/A", previous_idx,
            )
            return None

        alert_level = _ALERT_LEVEL_MAP.get(current, "INFO")
        threshold = RISK_THRESHOLDS.get(current, {})

        # Human-readable message
        if previous:
            message = (
                f"⚠️ Risk level ESCALATED from {previous.upper()} to "
                f"{current.upper()}. {threshold.get('icon', '')} "
                f"{threshold.get('label', current.title())} conditions detected."
            )
        else:
            message = (
                f"{threshold.get('icon', '')} Initial risk assessment: "
                f"{threshold.get('label', current.title())} conditions detected."
            )

        alert_id = f"ALT-{timestamp.strftime('%Y%m%d-%H%M%S')}-{current_idx:03d}"

        alert: Dict[str, Any] = {
            "alert_id": alert_id,
            "timestamp": timestamp.isoformat(),
            "alert_level": alert_level,
            "current_risk": current,
            "previous_risk": previous,
            "escalated": True,
            "message": message,
            "color": threshold.get("color", "#6b7280"),
            "icon": threshold.get("icon", "ℹ️"),
        }

        # Persist
        self._save_alert(alert)

        logger.warning(
            "ALERT [%s] %s: %s → %s | %s",
            alert_id, alert_level,
            previous or "N/A", current, message,
        )

        return alert

    def get_alert_history(self) -> List[Dict[str, Any]]:
        """Retrieve the full alert history from the JSON log.

        Returns:
            List of alert dictionaries, chronologically ordered.
        """
        self._ensure_log_file()
        try:
            raw = _ALERT_LOG_FILE.read_text(encoding="utf-8")
            history: list = json.loads(raw)
            logger.debug("Loaded %d alerts from history.", len(history))
            return history
        except (json.JSONDecodeError, IOError) as exc:
            logger.error(
                "Corrupted alert log; resetting. Error: %s", exc
            )
            _ALERT_LOG_FILE.write_text("[]", encoding="utf-8")
            return []

    def send_notification(self, alert: Dict[str, Any]) -> bool:
        """Dispatch an alert notification via email.

        If SMTP is not configured (``ALERT_EMAIL_ENABLED=false``), the
        notification is *simulated* – logged to the console but not
        actually sent.

        Args:
            alert: Alert dictionary as returned by :meth:`check_alert`.

        Returns:
            ``True`` if the notification was sent (or simulated)
            successfully, ``False`` on failure.
        """
        if not alert:
            logger.warning("send_notification called with empty alert.")
            return False

        subject = (
            f"[Epi Predict {alert.get('alert_level', 'ALERT')}] "
            f"Influenza Risk: {alert.get('current_risk', 'unknown').upper()}"
        )
        body = self._format_email_body(alert)

        if not ALERT_EMAIL_ENABLED:
            logger.info(
                "EMAIL SIMULATED (SMTP not configured):\n"
                "  To: %s\n  Subject: %s\n  Body:\n%s",
                ALERT_RECIPIENTS,
                subject,
                body,
            )
            return True

        # Real email dispatch
        try:
            recipients = [r.strip() for r in ALERT_RECIPIENTS if r.strip()]
            if not recipients:
                logger.warning("No alert recipients configured.")
                return False

            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = SMTP_USER
            msg["To"] = ", ".join(recipients)
            msg.attach(MIMEText(body, "plain"))
            msg.attach(MIMEText(self._format_email_html(alert), "html"))

            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.sendmail(SMTP_USER, recipients, msg.as_string())

            logger.info(
                "Email sent to %d recipient(s): %s", len(recipients), subject
            )
            return True

        except Exception as exc:
            logger.error(
                "Failed to send email notification: %s", exc, exc_info=True
            )
            return False

    def clear_alerts(self) -> None:
        """Clear all alerts from the history log."""
        _ALERT_LOG_FILE.write_text("[]", encoding="utf-8")
        logger.info("Alert history cleared.")

    # ── Email formatting ─────────────────────────────────────────────────

    @staticmethod
    def _format_email_body(alert: Dict[str, Any]) -> str:
        """Format alert as plain-text email body."""
        lines = [
            "=" * 60,
            f"  EPI PREDICT – {alert.get('alert_level', 'ALERT')} NOTIFICATION",
            "=" * 60,
            "",
            f"Alert ID   : {alert.get('alert_id', 'N/A')}",
            f"Timestamp  : {alert.get('timestamp', 'N/A')}",
            f"Alert Level: {alert.get('alert_level', 'N/A')}",
            f"Risk Change: {(alert.get('previous_risk') or 'N/A').upper()} → "
            f"{alert.get('current_risk', 'N/A').upper()}",
            "",
            f"Message    : {alert.get('message', '')}",
            "",
            "-" * 60,
            "This is an automated alert from the Epi Predict Influenza",
            "Outbreak Early Warning System.  Do not reply to this email.",
            "-" * 60,
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_email_html(alert: Dict[str, Any]) -> str:
        """Format alert as HTML email body."""
        color = alert.get("color", "#6b7280")
        icon = alert.get("icon", "ℹ️")
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: auto;">
            <div style="background: {color}; color: white; padding: 20px;
                        border-radius: 8px 8px 0 0; text-align: center;">
                <h1 style="margin: 0;">{icon} Epi Predict Alert</h1>
                <p style="margin: 5px 0 0; font-size: 18px;">
                    {alert.get('alert_level', 'ALERT')}
                </p>
            </div>
            <div style="padding: 20px; border: 1px solid #e5e7eb;
                        border-radius: 0 0 8px 8px;">
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Alert ID</td>
                        <td style="padding: 8px;">{alert.get('alert_id', 'N/A')}</td>
                    </tr>
                    <tr style="background: #f9fafb;">
                        <td style="padding: 8px; font-weight: bold;">Timestamp</td>
                        <td style="padding: 8px;">{alert.get('timestamp', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Risk Transition</td>
                        <td style="padding: 8px;">
                            {(alert.get('previous_risk') or 'N/A').upper()} →
                            <strong>{alert.get('current_risk', 'N/A').upper()}</strong>
                        </td>
                    </tr>
                </table>
                <p style="margin-top: 16px; padding: 12px; background: #fef3c7;
                          border-left: 4px solid {color}; border-radius: 4px;">
                    {alert.get('message', '')}
                </p>
                <hr style="margin: 20px 0; border: none; border-top: 1px solid #e5e7eb;">
                <p style="font-size: 12px; color: #6b7280; text-align: center;">
                    Automated alert from Epi Predict – Influenza Outbreak Early Warning System
                </p>
            </div>
        </body>
        </html>
        """

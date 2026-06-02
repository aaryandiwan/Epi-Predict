"""
Epi Predict – Risk Classification Module

Classifies predicted influenza case counts into discrete risk levels
(Low, Moderate, High, Severe) using configurable thresholds. Provides
single-value classification, batch processing, trend analysis, and
risk summary generation for dashboard consumption.

Risk Levels (from config.settings.RISK_THRESHOLDS):
    ┌───────────┬──────────────┬───────────────────┐
    │ Level     │ Range        │ Label             │
    ├───────────┼──────────────┼───────────────────┤
    │ low       │ 0 – 500      │ Low Risk          │
    │ moderate  │ 500 – 2,000  │ Moderate Risk     │
    │ high      │ 2,000 – 5,000│ High Risk         │
    │ severe    │ 5,000 +      │ Severe Outbreak   │
    └───────────┴──────────────┴───────────────────┘

Author : Epi Predict Team
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from config.settings import RISK_THRESHOLDS

logger = logging.getLogger("epi_predict.risk_classifier")

# ─── Internal Constants ──────────────────────────────────────────────────────

# Ordered from lowest to highest severity for normalisation
_ORDERED_LEVELS: List[str] = ["low", "moderate", "high", "severe"]

# Maximum finite boundary used for 0-1 score normalisation.  We cap at the
# upper boundary of "high" (5 000) so that *any* value ≥ 5 000 maps to 1.0.
_NORM_CEILING: float = max(
    t["max"] for t in RISK_THRESHOLDS.values() if t["max"] != float("inf")
)


# ─── Core Classification ────────────────────────────────────────────────────

def classify_risk(predicted_value: float) -> Dict[str, Any]:
    """Classify a single predicted case count into a risk level.

    The ``score`` field is a 0-1 normalised severity indicator computed as
    ``min(value / _NORM_CEILING, 1.0)`` so downstream consumers can render
    gradient-based visuals without knowing the raw thresholds.

    Args:
        predicted_value: Predicted influenza case count (non-negative float).

    Returns:
        Dictionary with keys:
            - **level** (str): Threshold key, e.g. ``"low"``.
            - **label** (str): Human-readable label, e.g. ``"Low Risk"``.
            - **color** (str): Hex colour code for UI rendering.
            - **icon** (str): Emoji icon for quick visual scanning.
            - **score** (float): 0-1 normalised severity score.
            - **value** (float): The original predicted value (echo-back).

    Raises:
        ValueError: If *predicted_value* is negative.

    Example::

        >>> classify_risk(1200)
        {'level': 'moderate', 'label': 'Moderate Risk', ...}
    """
    if predicted_value < 0:
        logger.warning(
            "Negative prediction (%s) received; clamping to 0.", predicted_value
        )
        predicted_value = 0.0

    predicted_value = float(predicted_value)

    for level in _ORDERED_LEVELS:
        threshold = RISK_THRESHOLDS[level]
        if threshold["min"] <= predicted_value < threshold["max"]:
            score = min(predicted_value / _NORM_CEILING, 1.0)
            result = {
                "level": level,
                "label": threshold["label"],
                "color": threshold["color"],
                "icon": threshold["icon"],
                "score": round(score, 4),
                "value": predicted_value,
            }
            logger.debug(
                "Classified value=%.2f → %s (score=%.4f)",
                predicted_value,
                level,
                score,
            )
            return result

    # Fallback – should only trigger if thresholds are misconfigured
    logger.error(
        "Value %.2f did not match any threshold; defaulting to severe.",
        predicted_value,
    )
    return {
        "level": "severe",
        "label": RISK_THRESHOLDS["severe"]["label"],
        "color": RISK_THRESHOLDS["severe"]["color"],
        "icon": RISK_THRESHOLDS["severe"]["icon"],
        "score": 1.0,
        "value": predicted_value,
    }


# ─── Batch Classification ───────────────────────────────────────────────────

def classify_batch(values: List[float]) -> List[Dict[str, Any]]:
    """Classify a list of predicted values.

    This is a convenience wrapper around :func:`classify_risk` that
    processes an iterable of predictions in one call.

    Args:
        values: List of predicted case counts.

    Returns:
        List of classification dictionaries (same schema as
        :func:`classify_risk`).
    """
    if not values:
        logger.warning("classify_batch called with empty list.")
        return []

    logger.info("Batch classifying %d values.", len(values))
    return [classify_risk(v) for v in values]


# ─── Risk Trend ──────────────────────────────────────────────────────────────

def get_risk_trend(historical_values: List[float]) -> List[Dict[str, Any]]:
    """Compute risk level for each entry in a historical time-series.

    Useful for rendering a colour-coded timeline on the dashboard.

    Args:
        historical_values: Chronologically ordered list of case counts.

    Returns:
        List of classification dicts augmented with an ``index`` key
        indicating position in the input list.
    """
    if not historical_values:
        logger.warning("get_risk_trend called with empty history.")
        return []

    logger.info("Computing risk trend for %d data points.", len(historical_values))
    trend: List[Dict[str, Any]] = []
    for idx, val in enumerate(historical_values):
        entry = classify_risk(val)
        entry["index"] = idx
        trend.append(entry)

    return trend


# ─── Risk Summary ────────────────────────────────────────────────────────────

def get_current_risk_summary(predictions: List[float]) -> Dict[str, Any]:
    """Generate an aggregate risk summary for a set of predictions.

    Typically called with the latest forecast window (e.g. 12-week
    predictions) to produce a single summary widget for the dashboard.

    Args:
        predictions: List of predicted case counts.

    Returns:
        Dictionary with keys:
            - **current_risk**: Classification of the *maximum* predicted
              value (worst-case for the forecast window).
            - **max_value** / **min_value** / **avg_value**: Basic statistics.
            - **max_risk**: The classification dict for the maximum value.
            - **avg_risk**: The classification dict for the average value.
            - **trend_direction**: ``"increasing"``, ``"decreasing"``, or
              ``"stable"`` based on the first-half vs second-half comparison.
            - **num_predictions**: Number of predictions analysed.
            - **risk_distribution**: Count of predictions in each risk level.
    """
    if not predictions:
        logger.warning("get_current_risk_summary called with empty predictions.")
        return {
            "current_risk": classify_risk(0),
            "max_value": 0.0,
            "min_value": 0.0,
            "avg_value": 0.0,
            "max_risk": classify_risk(0),
            "avg_risk": classify_risk(0),
            "trend_direction": "stable",
            "num_predictions": 0,
            "risk_distribution": {level: 0 for level in _ORDERED_LEVELS},
        }

    max_val = max(predictions)
    min_val = min(predictions)
    avg_val = sum(predictions) / len(predictions)

    # Trend detection: compare mean of first half vs second half
    mid = len(predictions) // 2
    if mid > 0:
        first_half_mean = sum(predictions[:mid]) / mid
        second_half_mean = sum(predictions[mid:]) / (len(predictions) - mid)
        diff_pct = (
            (second_half_mean - first_half_mean) / max(first_half_mean, 1.0)
        ) * 100

        if diff_pct > 10:
            trend_direction = "increasing"
        elif diff_pct < -10:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
    else:
        trend_direction = "stable"

    # Risk distribution
    classified = classify_batch(predictions)
    distribution: Dict[str, int] = {level: 0 for level in _ORDERED_LEVELS}
    for item in classified:
        distribution[item["level"]] += 1

    summary = {
        "current_risk": classify_risk(max_val),
        "max_value": round(max_val, 2),
        "min_value": round(min_val, 2),
        "avg_value": round(avg_val, 2),
        "max_risk": classify_risk(max_val),
        "avg_risk": classify_risk(avg_val),
        "trend_direction": trend_direction,
        "num_predictions": len(predictions),
        "risk_distribution": distribution,
    }

    logger.info(
        "Risk summary: max=%.2f (%s), avg=%.2f (%s), trend=%s",
        max_val,
        summary["max_risk"]["level"],
        avg_val,
        summary["avg_risk"]["level"],
        trend_direction,
    )

    return summary

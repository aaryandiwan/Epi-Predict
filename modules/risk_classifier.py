"""
Epi Predict – Risk Classification Module

Classifies predicted influenza case counts into discrete risk levels
(Low, Moderate, High, Severe) using dynamic, country-specific thresholds
based on historical percentiles.

Author : Epi Predict Team
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
import numpy as np

from config.settings import DYNAMIC_RISK_PERCENTILES

logger = logging.getLogger("epi_predict.risk_classifier")

# ─── Internal Constants ──────────────────────────────────────────────────────

_ORDERED_LEVELS: List[str] = ["low", "moderate", "high", "severe"]

RISK_UI = {
    "low": {"label": "Low Risk", "color": "#22c55e", "icon": "✅"},
    "moderate": {"label": "Moderate Risk", "color": "#f59e0b", "icon": "⚠️"},
    "high": {"label": "High Risk", "color": "#ef4444", "icon": "🔴"},
    "severe": {"label": "Severe Outbreak", "color": "#7c2d12", "icon": "🚨"},
}

def get_dynamic_thresholds(historical_cases: List[float]) -> Dict[str, Dict[str, Any]]:
    """Compute dynamic thresholds based on historical percentiles of non-zero cases."""
    non_zero = [v for v in historical_cases if v > 0]
    if not non_zero:
        non_zero = [10.0, 50.0, 100.0]  # Fallback if no history

    p50 = np.percentile(non_zero, DYNAMIC_RISK_PERCENTILES["low"] * 100)
    p75 = np.percentile(non_zero, DYNAMIC_RISK_PERCENTILES["moderate"] * 100)
    p90 = np.percentile(non_zero, DYNAMIC_RISK_PERCENTILES["high"] * 100)

    # Ensure uniqueness if distribution is highly skewed
    if p50 == 0: p50 = 10.0
    if p75 <= p50: p75 = p50 * 1.5
    if p90 <= p75: p90 = p75 * 1.5

    return {
        "low": {"min": 0, "max": p50, **RISK_UI["low"]},
        "moderate": {"min": p50, "max": p75, **RISK_UI["moderate"]},
        "high": {"min": p75, "max": p90, **RISK_UI["high"]},
        "severe": {"min": p90, "max": float("inf"), **RISK_UI["severe"]}
    }


def classify_risk(predicted_value: float, historical_cases: List[float]) -> Dict[str, Any]:
    """Classify a single predicted case count into a risk level using dynamic thresholds."""
    if predicted_value < 0:
        predicted_value = 0.0

    predicted_value = float(predicted_value)
    thresholds = get_dynamic_thresholds(historical_cases)
    norm_ceiling = thresholds["high"]["max"]

    for level in _ORDERED_LEVELS:
        t = thresholds[level]
        if t["min"] <= predicted_value < t["max"]:
            score = min(predicted_value / norm_ceiling, 1.0) if norm_ceiling > 0 else 0
            return {
                "level": level,
                "label": t["label"],
                "color": t["color"],
                "icon": t["icon"],
                "score": round(score, 4),
                "value": predicted_value,
                "thresholds": thresholds # Pass back so UI can explain
            }

    # Fallback to severe
    score = min(predicted_value / norm_ceiling, 1.0) if norm_ceiling > 0 else 0
    return {
        "level": "severe",
        "label": RISK_UI["severe"]["label"],
        "color": RISK_UI["severe"]["color"],
        "icon": RISK_UI["severe"]["icon"],
        "score": round(score, 4),
        "value": predicted_value,
        "thresholds": thresholds
    }


def classify_batch(values: List[float], historical_cases: List[float]) -> List[Dict[str, Any]]:
    """Classify a list of predicted values."""
    if not values:
        return []
    return [classify_risk(v, historical_cases) for v in values]


def get_risk_trend(historical_values: List[float]) -> List[Dict[str, Any]]:
    """Compute risk level for each entry in a historical time-series."""
    if not historical_values:
        return []
    
    # Use the full dataset to compute its own thresholds
    trend = []
    for idx, val in enumerate(historical_values):
        entry = classify_risk(val, historical_values)
        entry["index"] = idx
        trend.append(entry)
    return trend


def get_current_risk_summary(predictions: List[float], historical_cases: List[float]) -> Dict[str, Any]:
    """Generate an aggregate risk summary for a set of predictions."""
    if not predictions:
        return {
            "current_risk": classify_risk(0, [10.0]),
            "max_value": 0.0,
            "min_value": 0.0,
            "avg_value": 0.0,
            "max_risk": classify_risk(0, [10.0]),
            "avg_risk": classify_risk(0, [10.0]),
            "trend_direction": "stable",
            "num_predictions": 0,
            "risk_distribution": {level: 0 for level in _ORDERED_LEVELS},
        }

    max_val = max(predictions)
    min_val = min(predictions)
    avg_val = sum(predictions) / len(predictions)

    mid = len(predictions) // 2
    if mid > 0:
        first_half_mean = sum(predictions[:mid]) / mid
        second_half_mean = sum(predictions[mid:]) / (len(predictions) - mid)
        diff_pct = ((second_half_mean - first_half_mean) / max(first_half_mean, 1.0)) * 100

        if diff_pct > 10: trend_direction = "increasing"
        elif diff_pct < -10: trend_direction = "decreasing"
        else: trend_direction = "stable"
    else:
        trend_direction = "stable"

    classified = classify_batch(predictions, historical_cases)
    distribution: Dict[str, int] = {level: 0 for level in _ORDERED_LEVELS}
    for item in classified:
        distribution[item["level"]] += 1

    return {
        "current_risk": classify_risk(max_val, historical_cases),
        "max_value": round(max_val, 2),
        "min_value": round(min_val, 2),
        "avg_value": round(avg_val, 2),
        "max_risk": classify_risk(max_val, historical_cases),
        "avg_risk": classify_risk(avg_val, historical_cases),
        "trend_direction": trend_direction,
        "num_predictions": len(predictions),
        "risk_distribution": distribution,
    }

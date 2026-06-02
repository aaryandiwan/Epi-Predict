import logging
from typing import Dict, List

logger = logging.getLogger("epi_predict.recommendation_engine")

RECOMMENDATION_TIERS = {
    "low": {
        "level": "low",
        "label": "Low Risk",
        "urgency": "low",
        "actions": [
            "Maintain hygiene",
            "Regular exercise",
            "Stay hydrated",
            "Get adequate sleep"
        ],
        "summary": "Standard health precautions are sufficient."
    },
    "moderate": {
        "level": "moderate",
        "label": "Moderate Risk",
        "urgency": "medium",
        "actions": [
            "Wear masks in crowded places",
            "Increase hand sanitization",
            "Monitor symptoms",
            "Avoid touching face"
        ],
        "summary": "Increased vigilance is recommended, especially in public spaces."
    },
    "high": {
        "level": "high",
        "label": "High Risk",
        "urgency": "high",
        "actions": [
            "Avoid crowded events",
            "Vaccination recommended",
            "Stock essential medicines",
            "Limit public transport"
        ],
        "summary": "Proactive preventive measures are strongly advised to reduce exposure."
    },
    "severe": {
        "level": "severe",
        "label": "Severe Outbreak",
        "urgency": "critical",
        "actions": [
            "Remote work where possible",
            "Avoid unnecessary travel",
            "Consult healthcare providers",
            "Follow government advisories",
            "Emergency kit preparation"
        ],
        "summary": "Strict adherence to public health guidelines is critical."
    }
}


def get_recommendations(risk_level: str) -> Dict[str, any]:
    """Get recommendations based on a specific risk level."""
    normalized_level = risk_level.lower()
    if normalized_level not in RECOMMENDATION_TIERS:
        logger.warning(f"Unknown risk level '{risk_level}', defaulting to 'low'")
        normalized_level = "low"
        
    return RECOMMENDATION_TIERS[normalized_level]


def get_all_recommendations() -> Dict[str, Dict[str, any]]:
    """Get all recommendation tiers."""
    return RECOMMENDATION_TIERS

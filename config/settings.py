"""
Epi Predict – Centralized Application Configuration

All application settings managed via environment variables with sensible defaults.
Uses Pydantic BaseSettings for validation and .env file support.
"""

import os
from pathlib import Path
from typing import Optional

# ─── Base Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models" / "saved_models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─── Data Source ─────────────────────────────────────────────────────────────
WHO_FLUNET_API_URL = os.getenv(
    "WHO_FLUNET_API_URL",
    "https://xmart-api-public.who.int/FLUMART/VIW_FNT?$format=csv"
)
DATA_CACHE_TTL_HOURS = int(os.getenv("DATA_CACHE_TTL_HOURS", "24"))
LOCAL_DATA_FILE = RAW_DATA_DIR / "influenza_data.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "processed_influenza.csv"


# ─── Default Country & Region ───────────────────────────────────────────────
DEFAULT_COUNTRY = os.getenv("DEFAULT_COUNTRY", "India")
DEFAULT_WHO_REGION = os.getenv("DEFAULT_WHO_REGION", "SEAR")  # South-East Asia


# ─── Model Configuration ────────────────────────────────────────────────────
TARGET_COLUMN = "Target"  # INF_A + INF_B
FEATURE_COLUMNS = ["ISO_YEAR", "ISO_WEEK", "Month", "lag_1", "lag_2", "roll_3"]
ALL_FEATURES = [
    "ISO_YEAR", "ISO_WEEK", "Month",
    "lag_1", "lag_2", "lag_3",
    "roll_3", "roll_5", "positivity_rate"
]
TRAIN_TEST_SPLIT = 0.8
RANDOM_STATE = 42
FORECAST_WEEKS = int(os.getenv("FORECAST_WEEKS", "12"))


# ─── Model Hyperparameters (exact from notebooks) ───────────────────────────
MODEL_PARAMS = {
    "linear_regression": {},
    "random_forest": {
        "n_estimators": 50,
        "max_depth": 10,
        "random_state": RANDOM_STATE,
    },
    "tuned_random_forest": {
        "param_distributions": {
            "n_estimators": [50, 100],
            "max_depth": [5, 8, 10],
            "min_samples_split": [5, 10],
            "min_samples_leaf": [2, 4],
            "max_features": ["sqrt", 0.8],
        },
        "n_iter": 10,
        "cv_splits": 3,
        "scoring": "neg_mean_absolute_error",
    },
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
    },
    "lstm": {
        "epochs": 50,
        "batch_size": 32,
        "validation_split": 0.1,
    },
    "arima": {
        "order": (2, 1, 2),
    },
    "stacking_ensemble": {
        "meta_learner_alpha": 1.0,
        "base_rf_n_estimators": 100,
        "base_rf_max_depth": 8,
        "base_xgb_n_estimators": 200,
        "base_xgb_max_depth": 5,
        "base_xgb_learning_rate": 0.05,
        "cv_splits": 5,
    },
}


# ─── Risk Classification Thresholds ─────────────────────────────────────────
# Note: These are now computed dynamically per country based on historical percentiles.
# See modules/risk_classifier.py for calculation logic.
DYNAMIC_RISK_PERCENTILES = {
    "low": 0.50,      # Below 50th percentile
    "moderate": 0.75, # 50th to 75th percentile
    "high": 0.90,     # 75th to 90th percentile
    # severe is > 90th percentile
}

RISK_THRESHOLDS = {
    "low": {"label": "Low Risk", "color": "#22c55e", "icon": "✅"},
    "moderate": {"label": "Moderate Risk", "color": "#f59e0b", "icon": "⚠️"},
    "high": {"label": "High Risk", "color": "#ef4444", "icon": "🔴"},
    "severe": {"label": "Severe Outbreak", "color": "#7c2d12", "icon": "🚨"},
}


# ─── API Configuration ──────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_TITLE = "Epi Predict – Influenza Outbreak Early Warning System API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Production-ready REST API for influenza forecasting, risk classification, "
    "and outbreak early warning powered by 7 ML models."
)


# ─── Dashboard Configuration ────────────────────────────────────────────────
DASHBOARD_TITLE = "🦠 Epi Predict – Influenza Outbreak Early Warning System"
DASHBOARD_ICON = "🦠"
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))


# ─── MLOps Configuration ────────────────────────────────────────────────────
MODEL_REGISTRY_FILE = MODELS_DIR / "model_registry.json"
PREDICTION_LOG_DB = LOGS_DIR / "predictions.db"
MONITORING_LOG_FILE = LOGS_DIR / "monitoring.log"
MAX_MODEL_VERSIONS = int(os.getenv("MAX_MODEL_VERSIONS", "10"))


# ─── Alert Configuration ────────────────────────────────────────────────────
ALERT_EMAIL_ENABLED = os.getenv("ALERT_EMAIL_ENABLED", "false").lower() == "true"
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ALERT_RECIPIENTS = os.getenv("ALERT_RECIPIENTS", "").split(",")


# ─── Logging ─────────────────────────────────────────────────────────────────
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "epi_predict.log"),
    ],
)

logger = logging.getLogger("epi_predict")

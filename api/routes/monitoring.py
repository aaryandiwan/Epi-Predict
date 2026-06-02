import time
from fastapi import APIRouter, HTTPException
from datetime import datetime

from api.schemas.responses import HealthCheckResponse
from config.settings import API_VERSION
from models.predictor import PredictionEngine
from mlops.prediction_logger import PredictionLogger
from mlops.monitoring import get_system_metrics
from data.data_loader import PROCESSED_DATA_FILE, DATA_CACHE_TTL_HOURS

router = APIRouter(prefix="/api", tags=["Monitoring"])

# Store app start time for uptime calculation
START_TIME = time.time()


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Basic health check and uptime."""
    uptime = time.time() - START_TIME
    
    # Check if models are loaded/trained
    try:
        engine = PredictionEngine()
        models_loaded = bool(engine.registry)
    except Exception:
        models_loaded = False
        
    return HealthCheckResponse(
        timestamp=datetime.now().isoformat(),
        uptime_seconds=uptime,
        models_loaded=models_loaded,
        api_version=API_VERSION
    )


@router.get("/metrics")
async def get_metrics():
    """System performance metrics."""
    try:
        return get_system_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prediction-logs")
async def get_prediction_logs(limit: int = 50):
    """Get recent prediction logs."""
    try:
        logger = PredictionLogger()
        return {"logs": logger.get_recent_logs(limit=limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-status")
async def get_data_status():
    """Get information about the local data cache."""
    if not PROCESSED_DATA_FILE.exists():
        return {
            "status": "missing",
            "message": "Data file not found. Fetch required."
        }
        
    mtime = PROCESSED_DATA_FILE.stat().st_mtime
    age_hours = (time.time() - mtime) / 3600
    is_fresh = age_hours < DATA_CACHE_TTL_HOURS
    
    return {
        "status": "fresh" if is_fresh else "stale",
        "last_updated": datetime.fromtimestamp(mtime).isoformat(),
        "age_hours": round(age_hours, 2),
        "ttl_hours": DATA_CACHE_TTL_HOURS,
        "path": str(PROCESSED_DATA_FILE)
    }

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any

from api.schemas.responses import RiskLevelResponse, RecommendationResponse
from api.routes.predictions import get_dynamic_engine
from models.predictor import PredictionEngine
from data.data_loader import load_and_prepare
from modules.risk_classifier import classify_risk
from modules.recommendation_engine import get_recommendations
from modules.alert_system import AlertSystem

router = APIRouter(prefix="/api", tags=["Risk & Alerts"])


@router.get("/risk-level", response_model=RiskLevelResponse)
async def get_current_risk(country: str = "India"):
    """Get current risk classification based on next week's forecast."""
    try:
        df = load_and_prepare(country=country)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for country: {country}")
            
        engine = get_dynamic_engine(country)
        forecast_result = engine.forecast_future(df=df, weeks_ahead=1)
        pred_val = forecast_result["forecast"]["predicted_cases"][0]
        
        historical_cases = df["Target"].tolist() if "Target" in df.columns else []
        risk = classify_risk(pred_val, historical_cases)
        return RiskLevelResponse(**risk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations", response_model=RecommendationResponse)
async def get_current_recommendations(country: str = "India"):
    """Get recommendations based on current risk level."""
    try:
        df = load_and_prepare(country=country)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for country: {country}")
            
        engine = get_dynamic_engine(country)
        forecast_result = engine.forecast_future(df=df, weeks_ahead=1)
        pred_val = forecast_result["forecast"]["predicted_cases"][0]
        
        historical_cases = df["Target"].tolist() if "Target" in df.columns else []
        risk = classify_risk(pred_val, historical_cases)
        recs = get_recommendations(risk["level"])
        return RecommendationResponse(**recs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/{level}", response_model=RecommendationResponse)
async def get_level_recommendations(level: str):
    """Get recommendations for a specific risk level."""
    try:
        recs = get_recommendations(level)
        return RecommendationResponse(**recs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_alerts():
    """Get active alerts and alert history."""
    try:
        alert_system = AlertSystem()
        history = alert_system.get_alert_history()
        # For simplicity, returning history. Active alerts would be recent escalations.
        return {"alert_history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

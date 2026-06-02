from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any

from api.schemas.requests import PredictionRequest
from api.schemas.responses import PredictionResponse, ForecastResponse
from models.predictor import PredictionEngine
from data.data_loader import load_and_prepare
from modules.risk_classifier import classify_risk

router = APIRouter(prefix="/api", tags=["Predictions"])


def get_engine():
    engine = PredictionEngine()
    if not engine.registry:
        raise HTTPException(status_code=503, detail="Models not trained yet. Please train models first.")
    return engine


@router.post("/predict", response_model=PredictionResponse)
async def predict_single_week(request: PredictionRequest, engine: PredictionEngine = Depends(get_engine)):
    """Generate a single week prediction for the given country."""
    try:
        # Load recent data to use for lag features
        df = load_and_prepare(country=request.country)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for country: {request.country}")
            
        # For a single week, we forecast 1 week ahead
        forecast_result = engine.forecast_future(
            df=df,
            weeks_ahead=1,
            model_name=request.model_name
        )
        
        pred_val = forecast_result["forecast"]["predicted_cases"][0]
        lower = forecast_result["forecast"]["lower_bound"][0]
        upper = forecast_result["forecast"]["upper_bound"][0]
        
        risk = classify_risk(pred_val)
        
        return PredictionResponse(
            predicted_cases=pred_val,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=request.confidence_level,
            model_used=forecast_result["model_used"],
            risk_level=risk["level"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/forecast/{weeks}", response_model=ForecastResponse)
async def get_forecast(weeks: int, country: str = "India", model_name: str = None, engine: PredictionEngine = Depends(get_engine)):
    """Generate a multi-week forecast."""
    if weeks < 1 or weeks > 52:
        raise HTTPException(status_code=400, detail="Weeks must be between 1 and 52")
        
    try:
        df = load_and_prepare(country=country)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for country: {country}")
            
        result = engine.forecast_future(
            df=df,
            weeks_ahead=weeks,
            model_name=model_name
        )
        return ForecastResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/forecast-data")
async def get_forecast_data(country: str = "India", engine: PredictionEngine = Depends(get_engine)):
    """Get raw forecast data."""
    try:
        df = load_and_prepare(country=country)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for country: {country}")
            
        result = engine.forecast_future(df=df, weeks_ahead=12)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

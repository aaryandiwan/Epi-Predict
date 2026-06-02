from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any

from api.schemas.responses import ReportResponse
from api.routes.predictions import get_dynamic_engine
from models.predictor import PredictionEngine
from data.data_loader import load_and_prepare
from modules.report_generator import generate_weekly_report
from modules.risk_classifier import classify_risk
from modules.recommendation_engine import get_recommendations
from modules.public_awareness import (
    get_flu_info, 
    get_preventive_measures, 
    get_vaccination_guidance, 
    get_emergency_contacts
)

router = APIRouter(prefix="/api", tags=["Reports & Info"])


@router.get("/weekly-report", response_model=ReportResponse)
async def get_weekly_report(country: str = "India"):
    """Generate comprehensive weekly health report."""
    try:
        df = load_and_prepare(country=country)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for country: {country}")
            
        engine = get_dynamic_engine(country)
        forecast_result = engine.forecast_future(df=df, weeks_ahead=4)
        predictions = forecast_result["forecast"]["predicted_cases"]
        
        historical_cases = df["Target"].tolist() if "Target" in df.columns else []
        risk = classify_risk(predictions[0], historical_cases)
        recs = get_recommendations(risk["level"])
        
        report_data = generate_weekly_report(predictions, risk["level"], recs)
        
        # Simple text formatting for API response
        text = f"Epi Predict Weekly Report\n"
        text += f"Risk Level: {risk['label']}\n"
        text += f"Forecast (Next 4 weeks): {', '.join(f'{p:.0f}' for p in predictions)}\n"
        text += f"Recommendations: {', '.join(recs['actions'])}\n"
        
        return ReportResponse(
            report_text=text,
            report_data=report_data,
            generated_at=forecast_result["generated_at"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/public-awareness")
async def get_all_public_awareness():
    """Get all public health information."""
    try:
        return {
            "flu_info": get_flu_info(),
            "prevention": get_preventive_measures(),
            "vaccination": get_vaccination_guidance(),
            "emergency_contacts": get_emergency_contacts()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/public-awareness/symptoms")
async def get_symptoms():
    try:
        return get_flu_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/public-awareness/prevention")
async def get_prevention():
    try:
        return {"measures": get_preventive_measures()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/public-awareness/vaccination")
async def get_vaccination():
    try:
        return get_vaccination_guidance()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/public-awareness/emergency-contacts")
async def get_contacts():
    try:
        return get_emergency_contacts()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class PredictionResponse(BaseModel):
    predicted_cases: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    model_used: str
    risk_level: str


class ForecastResponse(BaseModel):
    forecast: Dict[str, List[Any]]
    model_used: str
    weeks_ahead: int
    generated_at: str


class RiskLevelResponse(BaseModel):
    level: str
    label: str
    color: str
    icon: str
    score: float


class ModelInfoResponse(BaseModel):
    name: str
    path: str
    metrics: Dict[str, float]
    features: List[str]
    timestamp: str
    version: int


class ModelComparisonResponse(BaseModel):
    comparison: Dict[str, Any]
    best_model: str


class RecommendationResponse(BaseModel):
    level: str
    label: str
    actions: List[str]
    urgency: str
    summary: str


class ReportResponse(BaseModel):
    report_text: str
    report_data: Dict[str, Any]
    generated_at: str


class AlertResponse(BaseModel):
    active_alerts: List[Dict[str, Any]]
    alert_history: List[Dict[str, Any]]


class HealthCheckResponse(BaseModel):
    status: str = "healthy"
    timestamp: str
    uptime_seconds: float
    models_loaded: bool
    api_version: str

from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    country: str = Field(default="India", description="Country to predict for")
    weeks_ahead: int = Field(default=12, ge=1, le=52, description="Number of weeks to forecast")
    model_name: Optional[str] = Field(default=None, description="Specific model to use (defaults to best)")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="Confidence interval level")


class ModelCompareRequest(BaseModel):
    country: str = Field(default="India", description="Country to compare models on")


class RetrainRequest(BaseModel):
    force: bool = Field(default=False, description="Force retrain even if data is unchanged")


class AlertConfigRequest(BaseModel):
    email: str = Field(..., description="Email to send alerts to")
    threshold: str = Field(default="high", description="Risk threshold to trigger alert (moderate, high, severe)")

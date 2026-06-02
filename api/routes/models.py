from fastapi import APIRouter, HTTPException, Depends
from typing import List

from api.schemas.responses import ModelInfoResponse, ModelComparisonResponse
from api.routes.predictions import get_engine
from models.predictor import PredictionEngine
from data.data_loader import load_and_prepare
from models.feature_engineering import generate_future_features
from modules.explainability import get_shap_analysis, get_feature_importance

router = APIRouter(prefix="/api/models", tags=["Models"])


@router.get("", response_model=List[ModelInfoResponse])
async def list_models(engine: PredictionEngine = Depends(get_engine)):
    """List all available trained models and their metrics."""
    models_info = []
    for name, info in engine.get_model_info().items():
        models_info.append(
            ModelInfoResponse(
                name=name,
                path=info.get("path", ""),
                metrics=info.get("metrics", {}),
                features=info.get("features", []),
                timestamp=info.get("timestamp", ""),
                version=info.get("version", 1)
            )
        )
    return models_info


@router.get("/best", response_model=ModelInfoResponse)
async def get_best_model(engine: PredictionEngine = Depends(get_engine)):
    """Get information about the best performing model."""
    best_name = engine.best_model_name
    if not best_name:
        raise HTTPException(status_code=404, detail="Best model not set")
        
    info = engine.get_model_info(best_name)
    return ModelInfoResponse(
        name=best_name,
        path=info.get("path", ""),
        metrics=info.get("metrics", {}),
        features=info.get("features", []),
        timestamp=info.get("timestamp", ""),
        version=info.get("version", 1)
    )


@router.get("/{name}/metrics")
async def get_model_metrics(name: str, engine: PredictionEngine = Depends(get_engine)):
    """Get metrics for a specific model."""
    info = engine.get_model_info(name)
    if not info:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    return info.get("metrics", {})


@router.get("/compare", response_model=ModelComparisonResponse)
async def compare_models(country: str = "India", engine: PredictionEngine = Depends(get_engine)):
    """Compare predictions across all models."""
    try:
        df = load_and_prepare(country=country)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for country: {country}")
            
        # Get a row of features to predict on
        future_df = generate_future_features(df, weeks_ahead=1)
        features = engine.registry[engine.best_model_name]["features"]
        X = future_df[[f for f in features if f in future_df.columns]].values
        
        comparison = engine.compare_models(X)
        best_model = comparison.pop("_best_model", "unknown")
        
        return ModelComparisonResponse(
            comparison=comparison,
            best_model=best_model
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/explainability/{model_name}")
async def explain_model(model_name: str, engine: PredictionEngine = Depends(get_engine)):
    """Get SHAP explainability and feature importance for a model."""
    try:
        # Check if model exists
        engine.load_model(model_name)
        
        analysis = get_shap_analysis(model_name)
        importance = get_feature_importance(model_name)
        
        return {
            "model": model_name,
            "shap_analysis": analysis,
            "feature_importance": importance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

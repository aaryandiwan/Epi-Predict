import sys
from pathlib import Path

# Add project root to path so imports work correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from config.settings import API_TITLE, API_VERSION, API_DESCRIPTION
from api.routes import predictions, risk, models, reports, monitoring
from models.predictor import PredictionEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events for the FastAPI application."""
    print("Starting up Epi Predict API...")
    # Pre-load prediction engine to cache models if available
    try:
        engine = PredictionEngine()
        if engine.registry:
            print(f"Loaded {len(engine.registry)-2} models from registry.")
            print(f"Best model: {engine.best_model_name}")
        else:
            print("WARNING: No models found in registry. Please train models.")
    except Exception as e:
        print(f"Error initializing PredictionEngine: {e}")
        
    yield
    print("Shutting down Epi Predict API...")


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "message": str(exc)},
    )


# Include routers
app.include_router(predictions.router)
app.include_router(risk.router)
app.include_router(models.router)
app.include_router(reports.router)
app.include_router(monitoring.router)


@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "app": API_TITLE,
        "version": API_VERSION,
        "status": "online",
        "docs_url": "/docs",
        "message": "Welcome to the Epi Predict API. Visit /docs for interactive documentation."
    }

if __name__ == "__main__":
    import uvicorn
    from config.settings import API_HOST, API_PORT
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)

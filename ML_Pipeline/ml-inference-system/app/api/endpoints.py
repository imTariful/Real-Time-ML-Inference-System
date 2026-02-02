from fastapi import APIRouter, Depends, HTTPException, Header, Request
from typing import Annotated, Optional, List, Dict
from app.schemas import PredictionRequest, PredictionResponse, HealthResponse
from app.services.inference_engine import inference_engine, MODELS
from app.core.config import settings
from app.core.metrics import ACTIVE_MODELS
import structlog

logger = structlog.get_logger()
router = APIRouter()

async def verify_auth_token(x_token: Annotated[Optional[str], Header()] = None):
    """Verify authentication token from request header."""
    try:
        if x_token is None or x_token != settings.AUTH_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid or missing auth token")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("auth_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/models", response_model=Dict)
async def list_models():
    """Get list of available models."""
    try:
        models_list = []
        for version, config in MODELS.items():
            models_list.append({
                "version": version,
                "name": config["name"],
                "model_id": config["model_id"]
            })
        return {
            "total": len(models_list),
            "models": models_list
        }
    except Exception as e:
        logger.error("list_models_error", error=str(e))
        return {"total": 0, "models": []}

@router.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_auth_token)])
async def predict_sentiment(request: PredictionRequest):
    """Predict sentiment for given texts."""
    try:
        return await inference_engine.predict(request)
    except Exception as e:
        logger.error("predict_error", error=str(e), exc_info=True)
        return PredictionResponse(
            request_id=request.id,
            model_version=request.model_version or "v1",
            results=[],
            latency_ms=0
        )

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check health status."""
    try:
        return HealthResponse(
            status="active",
            active_models=list(MODELS.keys())
        )
    except Exception as e:
        logger.error("health_check_error", error=str(e))
        return HealthResponse(status="degraded", active_models=[])

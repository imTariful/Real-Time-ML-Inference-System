import sys
import os

# Ensure we're using the correct path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from contextlib import asynccontextmanager
import structlog
from app.core.logging import setup_logging

from app.api.endpoints import router as api_router
from app.core.config import settings
from app.services.cache_service import cache_service

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        setup_logging()
        logger.info("startup")
        await cache_service.connect()
    except Exception as e:
        logger.error("startup_error", error=str(e), exc_info=True)
    
    yield
    
    try:
        logger.info("shutdown")
    except Exception as e:
        logger.error("shutdown_error", error=str(e))

app = FastAPI(
    title="Real-Time ML Inference System",
    description="Scalable Inference API with ONNX Runtime and Redis",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/metrics")
async def metrics():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

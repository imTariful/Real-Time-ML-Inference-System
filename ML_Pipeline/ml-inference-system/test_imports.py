#!/usr/bin/env python
import sys
sys.path.insert(0, '.')

print("Testing config import...")
from app.core.config import settings
print("✓ Config loaded")

print("Testing logging setup...")
from app.core.logging import setup_logging
print("✓ Logging module loaded")

print("Testing metrics...")
from app.core.metrics import REQUEST_COUNT
print("✓ Metrics loaded")

print("Testing model loader...")
from app.services.model_loader import model_loader
print("✓ Model loader loaded")

print("Testing cache service...")
from app.services.cache_service import cache_service
print("✓ Cache service loaded")

print("Testing inference engine...")
from app.services.inference_engine import inference_engine
print("✓ Inference engine loaded")

print("Testing schemas...")
from app.schemas import PredictionRequest, PredictionResponse
print("✓ Schemas loaded")

print("Testing API endpoints...")
from app.api.endpoints import router
print("✓ API endpoints loaded")

print("Testing FastAPI app...")
from app.main import app
print("✓ FastAPI app loaded successfully!")

print("\n✓✓✓ ALL MODULES LOADED SUCCESSFULLY ✓✓✓")

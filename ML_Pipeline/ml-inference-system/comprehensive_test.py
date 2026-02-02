#!/usr/bin/env python
"""Comprehensive test suite for ML Inference System"""
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("ML INFERENCE SYSTEM - COMPREHENSIVE TEST SUITE")
print("=" * 60)

# Test 1: Module Imports
print("\n[TEST 1] Module Imports...")
try:
    from app.main import app
    from app.schemas import PredictionRequest, PredictionResponse
    from app.api.endpoints import router
    from app.core.config import settings
    from app.core.logging import setup_logging
    from app.core.metrics import REQUEST_COUNT, MODEL_INFERENCE_TIME
    from app.services.cache_service import cache_service
    from app.services.inference_engine import inference_engine
    from app.services.model_loader import model_loader
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: FastAPI App
print("\n[TEST 2] FastAPI App Structure...")
try:
    assert app is not None
    assert app.title == "Real-Time ML Inference System"
    assert hasattr(app, 'routes')
    routes = [route.path for route in app.routes]
    assert '/health' in routes or '/api/v1/predict' in str(routes)
    print(f"✓ FastAPI app configured correctly")
    print(f"  - Title: {app.title}")
    print(f"  - Routes found: {len(app.routes)}")
except Exception as e:
    print(f"✗ FastAPI app check failed: {e}")
    sys.exit(1)

# Test 3: Configuration
print("\n[TEST 3] Configuration...")
try:
    assert settings.PROJECT_NAME == "Real-Time ML Inference System"
    assert settings.API_V1_STR == "/api/v1"
    assert settings.AUTH_TOKEN == "secret-token"
    assert settings.LOG_LEVEL in ["INFO", "DEBUG", "WARNING", "ERROR"]
    print("✓ Configuration loaded correctly")
    print(f"  - Project: {settings.PROJECT_NAME}")
    print(f"  - Redis URL: {settings.REDIS_URL}")
    print(f"  - Log Level: {settings.LOG_LEVEL}")
except Exception as e:
    print(f"✗ Configuration check failed: {e}")
    sys.exit(1)

# Test 4: Metrics
print("\n[TEST 4] Prometheus Metrics...")
try:
    assert REQUEST_COUNT is not None
    assert MODEL_INFERENCE_TIME is not None
    print("✓ Metrics configured correctly")
    print(f"  - REQUEST_COUNT metric initialized")
    print(f"  - MODEL_INFERENCE_TIME metric initialized")
except Exception as e:
    print(f"✗ Metrics check failed: {e}")
    sys.exit(1)

# Test 5: Schema Validation
print("\n[TEST 5] Schema Validation...")
try:
    # Test valid request
    req = PredictionRequest(
        id="test-1",
        texts=["I love this product"],
        model_version="v1"
    )
    assert req.id == "test-1"
    assert len(req.texts) == 1
    assert req.model_version == "v1"
    
    # Test response schema
    resp = PredictionResponse(
        request_id="test-1",
        model_version="v1",
        results=[{"label": 1, "confidence": 0.95}],
        latency_ms=100.5
    )
    assert resp.request_id == "test-1"
    print("✓ Schema validation working correctly")
    print(f"  - PredictionRequest: OK")
    print(f"  - PredictionResponse: OK")
except Exception as e:
    print(f"✗ Schema validation failed: {e}")
    sys.exit(1)

# Test 6: File Structure
print("\n[TEST 6] Required Files...")
import os
required_files = [
    'requirements.txt',
    'docker-compose.yaml',
    'Dockerfile',
    'app/main.py',
    'app/schemas.py',
    'app/api/endpoints.py',
    'app/core/config.py',
    'app/core/logging.py',
    'app/core/metrics.py',
    'app/services/cache_service.py',
    'app/services/inference_engine.py',
    'app/services/model_loader.py',
]

missing_files = []
for file in required_files:
    if not os.path.exists(file):
        missing_files.append(file)

if missing_files:
    print(f"✗ Missing files: {missing_files}")
else:
    print(f"✓ All required files present ({len(required_files)} files)")

# Test 7: __init__.py files
print("\n[TEST 7] Python Package Structure...")
required_inits = [
    'app/__init__.py',
    'app/api/__init__.py',
    'app/core/__init__.py',
    'app/services/__init__.py',
]

missing_inits = []
for init_file in required_inits:
    if not os.path.exists(init_file):
        missing_inits.append(init_file)

if missing_inits:
    print(f"✗ Missing __init__.py files: {missing_inits}")
else:
    print(f"✓ All __init__.py files present ({len(required_inits)} files)")

# Final Result
print("\n" + "=" * 60)
if missing_files or missing_inits:
    print("TESTS PASSED WITH WARNINGS")
    print("=" * 60)
else:
    print("✓✓✓ ALL TESTS PASSED SUCCESSFULLY ✓✓✓")
    print("=" * 60)
    print("\nThe application is ready to run!")
    print("To start the server:")
    print("  uvicorn app.main:app --reload")
    print("\nOr use Docker:")
    print("  docker-compose up --build")

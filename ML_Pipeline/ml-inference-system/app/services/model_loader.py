import onnxruntime as ort
import threading
import os
from app.core.config import settings
from app.core.metrics import MODEL_LOAD_TIME, ACTIVE_MODELS
import time
import structlog

logger = structlog.get_logger()

class ModelLoader:
    _models = {}
    _lock = threading.Lock()

    @classmethod
    def get_model(cls, path: str):
        """
        Thread-safe method to get or load an ONNX model.
        """
        if path not in cls._models:
            with cls._lock:
                # Check again inside lock
                if path not in cls._models:
                    cls._load_model(path)
        return cls._models[path]

    @classmethod
    def _load_model(cls, path: str):
        logger.info("loading_model", path=path)
        start_time = time.time()
        try:
            # Verify file exists
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")

            # Load ONNX session
            session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            
            cls._models[path] = session
            
            duration = time.time() - start_time
            model_name = path.split("/")[-2] # primitive extraction
            version = path.split("/")[-2]
            
            MODEL_LOAD_TIME.labels(model_name="sentiment", model_version=version).set(duration)
            ACTIVE_MODELS.labels(model_name="sentiment", version=version).inc()
            
            logger.info("model_loaded", path=path, duration=duration)
        except Exception as e:
            logger.error("model_load_failed", path=path, error=str(e))
            raise e

model_loader = ModelLoader()

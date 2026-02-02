import asyncio
import time
from typing import List
from app.services.model_loader import model_loader
from app.services.cache_service import cache_service
from app.schemas import PredictionRequest, PredictionResponse
from app.core.metrics import MODEL_INFERENCE_TIME, REQUEST_LATENCY
import structlog
from transformers import pipeline

logger = structlog.get_logger()

# Cache for sentiment pipelines
_pipelines = {}

MODELS = {
    "v1": {
        "name": "DistilBERT SST-2",
        "model_id": "distilbert-base-uncased-finetuned-sst-2-english"
    },
    "v2": {
        "name": "RoBERTa Twitter",
        "model_id": "cardiffnlp/twitter-roberta-base-sentiment"
    },
    "v3": {
        "name": "NLPTown Multilingual Sentiment",
        "model_id": "nlptown/bert-base-multilingual-uncased-sentiment",
        "notes": "Multilingual sentiment model (NLPTown)"
    },
    "v4": {
        "name": "TinyBERT",
        "model_id": "huawei-noah/TinyBERT_General_4L_312D"
    },
    "v5": {
        "name": "BERT Base Multilingual",
        "model_id": "bert-base-multilingual-cased"
    }
}

def get_pipeline(model_version: str):
    """Get or create a sentiment analysis pipeline for the model version"""
    if model_version not in MODELS:
        model_version = "v1"  # Default to v1
    
    if model_version not in _pipelines:
        model_id = MODELS[model_version]["model_id"]
        try:
            _pipelines[model_version] = pipeline(
                "sentiment-analysis",
                model=model_id,
                device=-1  # CPU
            )
            logger.info("pipeline_loaded", model_version=model_version, model_id=model_id)
        except Exception as e:
            logger.error("pipeline_load_failed", model_version=model_version, error=str(e))
            # Fallback to v1
            if model_version != "v1":
                return get_pipeline("v1")
            raise
    
    return _pipelines[model_version]

class InferenceEngine:
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Predict sentiment for given texts using specified model.
        """
        try:
            start_time = time.time()
            model_version = request.model_version or "v1"
            
            # Get the appropriate pipeline
            sentiment_pipe = get_pipeline(model_version)
            
            # Run predictions
            results = []
            for text in request.texts:
                try:
                    # Get prediction from model
                    prediction = sentiment_pipe(text, truncation=True)[0]
                    
                    label_name = prediction["label"]
                    confidence = float(prediction["score"])
                    
                    # Map label to numeric
                    if label_name.lower() in ["positive", "neg_pos"]:
                        label = 1
                        class_name = "positive"
                    elif label_name.lower() in ["negative", "neg"]:
                        label = 0
                        class_name = "negative"
                    else:
                        label = 1 if confidence > 0.5 else 0
                        class_name = "positive" if label == 1 else "negative"
                    
                    results.append({
                        "text": text,
                        "label": label,
                        "confidence": confidence,
                        "class": class_name,
                        "raw_label": label_name
                    })
                except Exception as e:
                    logger.error("text_prediction_failed", text=text, error=str(e))
                    results.append({
                        "text": text,
                        "label": 0,
                        "confidence": 0.0,
                        "class": "unknown",
                        "error": str(e)
                    })
            
            duration_ms = (time.time() - start_time) * 1000
            MODEL_INFERENCE_TIME.labels("sentiment", model_version).observe(duration_ms / 1000)
            
            return PredictionResponse(
                request_id=request.id,
                model_version=model_version,
                results=results,
                latency_ms=duration_ms
            )
        except Exception as e:
            logger.error("predict_error", error=str(e), exc_info=True)
            return PredictionResponse(
                request_id=request.id,
                model_version=request.model_version or "v1",
                results=[],
                latency_ms=0
            )


inference_engine = InferenceEngine()

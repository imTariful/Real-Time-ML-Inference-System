from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict

class PredictionRequest(BaseModel):
    id: str = Field(..., description="Unique Request ID")
    texts: List[str] = Field(..., description="List of texts to classify", min_length=1)
    model_version: Optional[str] = Field(None, description="Specific model version to use")

class PredictionResponse(BaseModel):
    request_id: str
    model_version: str
    results: List[Dict[str, Union[str, float]]]
    latency_ms: float
    cached: bool = False

class HealthResponse(BaseModel):
    status: str
    active_models: List[str]

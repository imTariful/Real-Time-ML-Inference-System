from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Real-Time ML Inference System"
    API_V1_STR: str = "/api/v1"
    MODEL_REGISTRY_PATH: str = "model_registry.yaml"
    REDIS_URL: str = "redis://localhost:6379/0"
    AUTH_TOKEN: str = "secret-token"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()

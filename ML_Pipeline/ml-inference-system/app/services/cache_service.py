import redis.asyncio as redis
from app.core.config import settings
from app.core.metrics import CACHE_HITS, CACHE_MISSES
import hashlib
import json
import structlog

logger = structlog.get_logger()

class CacheService:
    def __init__(self):
        self.redis = None
    
    async def connect(self):
        try:
            self.redis = redis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
            await self.redis.ping()
            logger.info("connected_to_redis", url=settings.REDIS_URL)
        except Exception as e:
            logger.error("redis_connection_failed", error=str(e))
            self.redis = None

    async def get_prediction(self, model_version: str, input_text: str):
        if not self.redis:
            return None
        
        key = self._generate_key(model_version, input_text)
        try:
            cached = await self.redis.get(key)
            if cached:
                CACHE_HITS.labels(model_version=model_version).inc()
                return json.loads(cached)
            CACHE_MISSES.labels(model_version=model_version).inc()
        except Exception as e:
            logger.warn("cache_get_failed", error=str(e))
        return None

    async def set_prediction(self, model_version: str, input_text: str, result: dict, ttl: int = 3600):
        if not self.redis:
            return
        
        key = self._generate_key(model_version, input_text)
        try:
            await self.redis.set(key, json.dumps(result), ex=ttl)
        except Exception as e:
            logger.warn("cache_set_failed", error=str(e))

    def _generate_key(self, version: str, text: str) -> str:
        # Use MD5 for simple hashing of input text
        h = hashlib.md5(text.encode()).hexdigest()
        return f"pred:{version}:{h}"

cache_service = CacheService()

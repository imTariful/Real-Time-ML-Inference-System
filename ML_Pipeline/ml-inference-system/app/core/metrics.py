from prometheus_client import Counter, Histogram, Gauge

# Standard Metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
)

# ML Specific Metrics
MODEL_INFERENCE_TIME = Histogram(
    "model_inference_seconds",
    "Time taken for model inference",
    ["model_name", "model_version"],
    buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5]
)

MODEL_LOAD_TIME = Gauge(
    "model_load_seconds",
    "Time taken to load the model into memory",
    ["model_name", "model_version"]
)

ACTIVE_MODELS = Gauge(
    "active_models_count",
    "Number of models currently loaded",
    ["model_name", "version"]
)

CACHE_HITS = Counter(
    "cache_hits_total",
    "Total number of cache hits",
    ["model_version"]
)

CACHE_MISSES = Counter(
    "cache_misses_total",
    "Total number of cache misses",
    ["model_version"]
)

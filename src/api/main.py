"""
FastAPI application — ML model এর REST API।
Features: prediction endpoint, health check, Redis caching, Prometheus metrics।
"""
import hashlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager

import redis
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.api.schemas import PredictionRequest, PredictionResponse, HealthResponse
from src.models.predict import predict
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Prometheus Metrics ──────────────────────────────────
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"]
)
PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total predictions made",
    ["result", "risk_level"]
)
REQUEST_LATENCY = Histogram(
    "request_duration_seconds",
    "Request duration",
    ["endpoint"]
)
CACHE_HITS = Counter("cache_hits_total", "Redis cache hits")
CACHE_MISSES = Counter("cache_misses_total", "Redis cache misses")


# ── Redis Connection ────────────────────────────────────
def get_redis_client():
    try:
        client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD", None) or None,
            decode_responses=True,
        )
        client.ping()
        logger.info("✅ Redis connected")
        return client
    except Exception as e:
        logger.warning(f"Redis unavailable: {e}. Running without cache.")
        return None


redis_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App startup ও shutdown।"""
    global redis_client
    redis_client = get_redis_client()
    logger.info("🚀 FastAPI starting up...")
    yield
    logger.info("👋 FastAPI shutting down...")


# ── FastAPI App ─────────────────────────────────────────
app = FastAPI(
    title="Predictive Maintenance API",
    description="Industrial equipment failure prediction using ML",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """প্রতিটা request এর latency ও count track করো।"""
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()
    return response


# ── Endpoints ───────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    return {"message": "Predictive Maintenance API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """API health check।"""
    redis_ok = False
    if redis_client:
        try:
            redis_client.ping()
            redis_ok = True
        except Exception:
            pass
    return HealthResponse(status="healthy", redis=redis_ok)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_endpoint(request: PredictionRequest):
    """Machine failure prediction।

    Input: sensor readings (temperature, torque, speed, wear)
    Output: failure prediction (0/1), probability, risk level
    """
    data = request.model_dump()

    # Cache key তৈরি করো (input এর hash)
    cache_key = "pred:" + hashlib.md5(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()

    # Cache check করো
    if redis_client:
        cached = redis_client.get(cache_key)
        if cached:
            CACHE_HITS.inc()
            result = json.loads(cached)
            result["cached"] = True
            return PredictionResponse(**result)
        CACHE_MISSES.inc()

    # Predict করো
    try:
        result = predict(data)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Metrics track করো
    PREDICTION_COUNT.labels(
        result=str(result["prediction"]),
        risk_level=result["risk_level"],
    ).inc()

    # Cache এ save করো (১ ঘন্টা)
    if redis_client:
        redis_client.setex(cache_key, 3600, json.dumps(result))

    result["cached"] = False
    return PredictionResponse(**result)


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint।"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Current production model এর info।"""
    return {
        "model_name": os.getenv("MODEL_NAME", "predictive_maintenance_model"),
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
        "stage": "Production",
    }


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=False,
    )
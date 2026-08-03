from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry

from middleware.classifier import InjectionClassifier
from middleware.config import settings

try:
    from scripts.review_queue import ReviewQueue
except ImportError:
    ReviewQueue = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("middleware")

_registry = CollectorRegistry()
REQUEST_COUNT = Counter("injection_requests_total", "Total requests processed", registry=_registry)
BLOCKED_COUNT = Counter("injection_blocked_total", "Hard-blocked injection attempts", registry=_registry)
FLAGGED_COUNT = Counter("injection_flagged_total", "Soft-flagged injection attempts", registry=_registry)
CONFIDENCE = Histogram(
    "injection_confidence",
    "Confidence score distribution",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0],
    registry=_registry,
)
LATENCY = Histogram(
    "injection_latency_seconds",
    "Inference latency in seconds",
    buckets=[0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0],
    registry=_registry,
)
SHADOW_AGREEMENT = Counter("injection_shadow_agreements_total", "Shadow-production agreement count", registry=_registry)
SHADOW_DISAGREEMENT = Counter("injection_shadow_disagreements_total", "Shadow-production disagreement count", registry=_registry)
SHADOW_CONFIDENCE_DELTA = Histogram(
    "injection_shadow_confidence_delta",
    "Absolute confidence difference between shadow and production",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
    registry=_registry,
)


def extract_text(messages: list[dict]) -> str:
    texts = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text", ""))
            content = " ".join(parts)
        texts.append(str(content))
    return "\n".join(texts)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting middleware...")
    logger.info("Mode: %s | Threshold: %.2f", settings.mode, settings.threshold)
    logger.info("Model path: %s", settings.model_path)

    app.state.shadow_classifier = None
    app.state.review_queue = None
    app.state.request_id_counter = 0

    classifier = InjectionClassifier(
        model_path=settings.model_path,
        max_length=settings.max_length,
    )
    app.state.classifier = classifier

    if settings.shadow_enabled and settings.shadow_model_path:
        logger.info("Shadow mode enabled — loading shadow model from: %s", settings.shadow_model_path)
        app.state.shadow_classifier = InjectionClassifier(
            model_path=settings.shadow_model_path,
            max_length=settings.max_length,
        )

    if ReviewQueue is not None:
        app.state.review_queue = ReviewQueue(settings.review_queue_path)
        logger.info("Review queue ready at: %s", settings.review_queue_path)

    app.state.httpx_client = httpx.AsyncClient(timeout=60.0)
    logger.info("Middleware ready on %s:%s", settings.host, settings.port)
    yield
    await app.state.httpx_client.aclose()


app = FastAPI(title="Prompt Injection Detector", lifespan=lifespan)


def get_classifier() -> InjectionClassifier:
    return app.state.classifier


@app.get("/health")
async def health():
    classifier = get_classifier()
    return {
        "status": "ok" if classifier.is_ready else "starting",
        "model_loaded": classifier.is_ready,
        "mode": settings.mode,
        "threshold": settings.threshold,
    }


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(_registry), media_type="text/plain")


@app.post("/chat/completions")
async def chat_completion(request: Request):
    classifier = get_classifier()
    if not classifier.is_ready:
        return JSONResponse(status_code=503, content={"error": "Model not ready"})

    body = await request.json()
    messages = body.get("messages", [])

    if not messages:
        return JSONResponse(status_code=400, content={"error": "No messages provided"})

    text = extract_text(messages)

    app.state.request_id_counter += 1
    request_id = app.state.request_id_counter

    start = time.perf_counter()
    label, confidence = classifier.predict(text)
    elapsed = time.perf_counter() - start

    REQUEST_COUNT.inc()
    CONFIDENCE.observe(confidence)
    LATENCY.observe(elapsed)

    logger.debug("Prediction: %s | confidence: %.4f | latency: %.3fs",
                 "INJECTION" if label == 1 else "BENIGN", confidence, elapsed)

    shadow_label, shadow_confidence = None, None
    if app.state.shadow_classifier is not None:
        shadow_label, shadow_confidence = app.state.shadow_classifier.predict(text)
        delta = abs(confidence - shadow_confidence)
        SHADOW_CONFIDENCE_DELTA.observe(delta)
        if shadow_label == label:
            SHADOW_AGREEMENT.inc()
        else:
            SHADOW_DISAGREEMENT.inc()

    review_queue = app.state.review_queue
    if review_queue is not None:
        review_queue.add(
            request_id=request_id,
            text=text,
            production_pred=label,
            production_conf=confidence,
            shadow_pred=shadow_label,
            shadow_conf=shadow_confidence,
        )

    if label == 1 and confidence >= settings.threshold:
        if settings.mode == "hard_block":
            BLOCKED_COUNT.inc()
            logger.info("BLOCKED injection (conf=%.4f)", confidence)
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Prompt injection detected",
                    "confidence": round(confidence, 4),
                    "injection_detected": True,
                },
            )
        else:
            FLAGGED_COUNT.inc()
            logger.info("FLAGGED injection (conf=%.4f)", confidence)

    forward_headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = await app.state.httpx_client.post(
            f"{settings.llm_endpoint}/chat/completions",
            json=body,
            headers=forward_headers,
        )
    except httpx.RequestError as e:
        logger.error("LLM endpoint unreachable: %s", e)
        return JSONResponse(status_code=502, content={"error": "LLM endpoint unreachable"})

    resp_headers = dict(response.headers)
    if label == 1 and confidence >= settings.threshold and settings.mode == "soft_flag":
        resp_headers["X-Injection-Detected"] = str(round(confidence, 4))

    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=resp_headers,
        media_type=response.headers.get("content-type", "application/json"),
    )


def main():
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()

from dotenv import load_dotenv
import os

# Load .env from the same directory as this file (app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))


import time
import uuid
from collections import defaultdict, deque
from typing import Optional, Dict, Any, List, Deque, Tuple

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.deps import PROJECT_ROOT  # noqa: F401
from api.schemas import TopicResponse, EvaluateTextRequest, EvaluateTextResponse
from api.security import require_auth, authorize_request_for_docs

from topic_generation.generate_topic import get_random_topic, list_categories, search_topics
from speech_analysis.stage3_analysis import analyze_fillers, calculate_wpm, analyze_grammar
from scoring_feedback.stage4_scoring import run_stage4
from topic_relevance.stage5_relevance import run_stage5
from coaching.stage6_coaching import run_stage6


app = FastAPI(
    title="GARGI API",
    version="0.1",
    description="Guided AI for Real-world General Interaction — FastAPI Layer"
)

# CORS (MVP-friendly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later: restrict to your app domain
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Simple in-memory rate limiter (MVP)
# ----------------------------
# key -> timestamps (seconds). Keep small & safe for LAN.
_RATE_WINDOW_SEC = 60
_RATE_MAX_REQ = 30  # per minute per client (tune as needed)
_rate_buckets: Dict[str, Deque[float]] = defaultdict(deque)

def _client_key(request: Request) -> str:
    # Prefer explicit user_id header later; for now use client host
    host = request.client.host if request.client else "unknown"
    return host

def _enforce_rate_limit(request: Request) -> None:
    key = _client_key(request)
    now = time.time()
    bucket = _rate_buckets[key]

    # drop old
    cutoff = now - _RATE_WINDOW_SEC
    while bucket and bucket[0] < cutoff:
        bucket.popleft()

    if len(bucket) >= _RATE_MAX_REQ:
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")
    bucket.append(now)

# ----------------------------
# Middleware: docs protection + request-id + timing + rate limit
# ----------------------------
@app.middleware("http")
async def gargi_request_middleware(request: Request, call_next):
    # Public endpoints
    if request.url.path in ["/health", "/"]:
        return await call_next(request)

    # Rate limit everything except health/root
    try:
        _enforce_rate_limit(request)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})

    # Protect docs + openapi
    if request.url.path.startswith("/docs") or request.url.path.startswith("/redoc") or request.url.path == "/openapi.json":
        try:
            authorize_request_for_docs(request)
        except HTTPException as e:
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail},
                headers=getattr(e, "headers", None) or {},
            )

    # Request ID + timing
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start = time.time()
    try:
        response = await call_next(request)
    finally:
        elapsed_ms = int((time.time() - start) * 1000)
        request.state.elapsed_ms = elapsed_ms

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Elapsed-ms"] = str(request.state.elapsed_ms)
    return response


# ----------------------------
# Global error handling (Android-friendly)
# ----------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    rid = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "ok": False,
            "request_id": rid,
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail if isinstance(exc.detail, str) else str(exc.detail),
            },
        },
        headers=getattr(exc, "headers", None) or {},
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "request_id": rid,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred.",
            },
        },
    )


# ----------------------------
# Helpers
# ----------------------------
def _resolve_user_id(payload: EvaluateTextRequest, request: Request) -> str:
    uid = (payload.user_id or "").strip()
    if uid:
        return uid
    # MVP: auto-assign guest id per client host
    host = request.client.host if request.client else "unknown"
    return f"guest_{host}"


def normalize_topic_obj(topic_obj: Optional[Dict[str, Any]], topic_text: Optional[str]) -> Dict[str, Any]:
    if topic_obj and isinstance(topic_obj, dict):
        if not topic_obj.get("topic_raw"):
            t = topic_obj.get("topic") or topic_obj.get("topic_content") or topic_obj.get("topic_text")
            if t:
                topic_obj["topic_raw"] = t

        if not topic_obj.get("topic_content"):
            topic_obj["topic_content"] = topic_obj.get("topic_raw", "")

        topic_obj.setdefault("topic_type", "general")
        topic_obj.setdefault("expected_anchors", [])
        topic_obj.setdefault("topic_keyphrases", [])
        return topic_obj

    if topic_text and topic_text.strip():
        t = topic_text.strip()
        return {
            "topic_raw": t,
            "topic_content": t,
            "topic_type": "general",
            "expected_anchors": [],
            "topic_keyphrases": [],
        }

    raise HTTPException(status_code=400, detail="Provide either topic_obj or topic_text.")


def topic_text_from_obj(topic_obj: Dict[str, Any]) -> str:
    return (topic_obj.get("topic_raw") or topic_obj.get("topic_content") or "").strip()


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {
        "service": "gargi-api",
        "status": "ok",
        "health": "/health",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }

@app.get("/health")
def health():
    return {"status": "ok", "service": "gargi-api", "version": "0.1"}


@app.get("/categories", dependencies=[Depends(require_auth)])
def categories() -> Dict[str, List[str]]:
    return {"categories": list_categories()}


@app.get("/topics", response_model=TopicResponse, dependencies=[Depends(require_auth)])
def topics(category: Optional[str] = None):
    try:
        topic_obj = get_random_topic(category=category)
        if not isinstance(topic_obj, dict):
            raise ValueError("Topic generator returned non-dict.")
        topic_obj = normalize_topic_obj(topic_obj, None)
        t = topic_text_from_obj(topic_obj)
        if not t:
            raise ValueError("Topic text missing in topic_obj.")
        return TopicResponse(topic_obj=topic_obj, topic_text=t)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Topic generation failed: {e}")


@app.get("/topics/search", dependencies=[Depends(require_auth)])
def topics_search(q: str, category: Optional[str] = None, limit: int = 10):
    q = (q or "").strip()
    if len(q) < 2:
        return {"results": []}
    limit = max(1, min(int(limit), 20))
    results = search_topics(query=q, category=category, limit=limit)
    return {"results": results}


@app.post("/evaluate/text", response_model=EvaluateTextResponse, dependencies=[Depends(require_auth)])
def evaluate_text(payload: EvaluateTextRequest, request: Request):
    transcript = (payload.transcript or "").strip()
    if len(transcript) < 3:
        raise HTTPException(status_code=400, detail="Transcript is empty or too short.")

    topic_obj = normalize_topic_obj(payload.topic_obj, payload.topic_text)
    topic_text = topic_text_from_obj(topic_obj)

    duration_sec = float(payload.duration_sec) if payload.duration_sec else 60.0

    stage3 = {
        "fluency": {
            "duration_sec": round(duration_sec, 2),
            "wpm": calculate_wpm(transcript, duration_sec),
            "pause_ratio": 0.0,
            "filler_words": analyze_fillers(transcript),
        },
        "grammar": analyze_grammar(transcript),
    }

    stage4 = run_stage4(stage3)
    stage5 = run_stage5(topic_obj=topic_obj, transcript=transcript)

    user_id = _resolve_user_id(payload, request)

    stage6 = run_stage6(
        topic_text=topic_text,
        transcript=transcript,
        stage4_results=stage4,
        stage5_results=stage5,
        save_history=bool(payload.save_history),
        user_id=user_id,
    )

    return EvaluateTextResponse(
        topic_obj=topic_obj,
        topic_text=topic_text,
        transcript=transcript,
        stage3=stage3,
        stage4=stage4,
        stage5=stage5,
        stage6=stage6,
        user_id=user_id,
    )

# api/app.py
import os
from pathlib import Path

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

# Load .env early (before importing modules that read env vars)
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if load_dotenv and ENV_PATH.exists():
    load_dotenv(dotenv_path=str(ENV_PATH), override=False)

from api.security import require_auth  # noqa: E402


def create_app() -> FastAPI:
    app = FastAPI(
        title="GARGI API",
        version="0.1",
    )

    # CORS (safe for LAN dev; tighten later)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------
    # ROUTES
    # -----------------------------
    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "gargi-api", "version": "0.1"}

    # If you already have routers, include them here.
    # Example:
    # from api.routes.topics import router as topics_router
    # app.include_router(topics_router)

    # Minimal protected endpoints to confirm auth works.
    # Replace these with your real implementations if they exist elsewhere.
    @app.get("/categories", dependencies=[Depends(require_auth)])
    async def categories():
        # If your real app loads categories from CSV/DB, call that here.
        # This minimal placeholder prevents startup breakage.
        return {
            "categories": [
                "Art & Literature", "Business", "Culture", "Current Affairs",
                "Education", "Entertainment", "Environment", "Food", "Health & Fitness",
                "History", "Lifestyle", "Motivation", "Personal", "Relationships",
                "Science", "Social Issues", "Sports", "Technology", "Travel"
            ]
        }

    @app.get("/topics", dependencies=[Depends(require_auth)])
    async def topics(category: str | None = None):
        # Placeholder response matching what your Android already logs.
        return {
            "topic_obj": {
                "topic_id": 1,
                "category": category or "General",
                "topic_raw": "Discuss a topic of your choice",
                "instruction": "discuss",
                "topic_content": "a topic of your choice",
                "topic_type": "general",
                "constraints": [],
                "expected_anchors": ["main_point", "supporting_points", "example"],
                "topic_keyphrases": ["topic", "choice"]
            },
            "topic_text": "Discuss a topic of your choice"
        }

    @app.post("/evaluate/text", dependencies=[Depends(require_auth)])
    async def evaluate_text(payload: dict):
        # Temporary safe echo until your real pipeline is wired correctly.
        # This prevents 500s while we align run_stage6 signature.
        transcript = payload.get("transcript", "")
        topic_text = payload.get("topic_text")
        duration_sec = payload.get("duration_sec")
        return {
            "result": "ok",
            "received": {
                "transcript_len": len(transcript),
                "topic_text": topic_text,
                "duration_sec": duration_sec
            }
        }

    return app


app = create_app()

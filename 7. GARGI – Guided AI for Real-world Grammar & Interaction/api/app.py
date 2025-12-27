# api/app.py
import os
from pathlib import Path

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# -------------------------------------------------
# 1) Load .env BEFORE importing any routers/security
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../api/.. (project root)
ENV_PATH = PROJECT_ROOT / ".env"

# If your .env is elsewhere, update ENV_PATH to that absolute path.
load_dotenv(dotenv_path=str(ENV_PATH), override=True)

# Optional: print to verify correct env in the uvicorn console
_api_key = (os.getenv("GARGI_API_KEY", "") or "").strip()
print(f"ENV_PATH: {ENV_PATH}")
print(f"GARGI_API_KEY prefix: {_api_key[:6]} len: {len(_api_key)}")

# Now safe to import security + routers
from api.security import require_auth, authorize_request_for_docs  # noqa: E402

# If you have routers like api.routes.xxx, import them here after dotenv
# from api.routes.topics import router as topics_router  # noqa: E402
# from api.routes.evaluate import router as evaluate_router  # noqa: E402


app = FastAPI(
    title="GARGI API",
    version="0.1",
)

# CORS (adjust if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# 2) Health endpoint stays open (no auth)
# -------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "service": "gargi-api", "version": "0.1"}


# -------------------------------------------------
# 3) Protect docs with Basic ONLY (optional)
# -------------------------------------------------
@app.get("/docs-auth", include_in_schema=False)
def docs_auth(_=Depends(authorize_request_for_docs)):
    return {"ok": True}


# -------------------------------------------------
# 4) Example protected endpoints (require_auth = API key OR Basic)
# -------------------------------------------------
@app.get("/categories")
def categories(_=Depends(require_auth)):
    # Replace with your real categories loader
    return {
        "categories": [
            "Art & Literature", "Business", "Culture", "Current Affairs", "Education",
            "Entertainment", "Environment", "Food", "Health & Fitness", "History",
            "Lifestyle", "Motivation", "Personal", "Relationships", "Science",
            "Social Issues", "Sports", "Technology", "Travel"
        ]
    }

from api.evaluate import router as evaluate_router

app.include_router(evaluate_router, dependencies=[Depends(require_auth)])

@app.get("/topics")
def topics(category: str | None = None, _=Depends(require_auth)):
    # Replace with your real topic provider
    return {"topic_obj": {}, "topic_text": "Example topic"}





# If you already have routers, use:
# app.include_router(topics_router, dependencies=[Depends(require_auth)])
# app.include_router(evaluate_router, dependencies=[Depends(require_auth)])

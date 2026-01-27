# D:\Machine_Learning_Projects\7. GARGI – Guided AI for Real-world Grammar & Interaction\api\app.py
import os
from pathlib import Path

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# -------------------------------------------------
# 1) Load .env BEFORE importing any routers/security
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # project root
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=str(ENV_PATH), override=True)

_api_key = (os.getenv("GARGI_API_KEY", "") or "").strip()
print(f"ENV_PATH: {ENV_PATH}")
print(f"GARGI_API_KEY prefix: {_api_key[:6]} len: {len(_api_key)}")

# Now safe to import auth
from api.security import require_auth, authorize_request_for_docs  # noqa: E402

# Routers
from api.evaluate import router as evaluate_router  # noqa: E402
from api.routes.topics import router as topics_router  # noqa: E402


app = FastAPI(
    title="GARGI API",
    version="0.1",
)

# CORS (LAN testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Public endpoint
# -------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "service": "gargi-api", "version": "0.1"}


# -------------------------------------------------
# Optional: protect docs with Basic only
# -------------------------------------------------
@app.get("/docs-auth", include_in_schema=False)
def docs_auth(_=Depends(authorize_request_for_docs)):
    return {"ok": True}


# -------------------------------------------------
# Protected routers (API key OR Basic)
# -------------------------------------------------
# IMPORTANT:
# - /health remains public
# - everything else is protected via require_auth

app.include_router(topics_router, dependencies=[Depends(require_auth)])
app.include_router(evaluate_router, dependencies=[Depends(require_auth)])

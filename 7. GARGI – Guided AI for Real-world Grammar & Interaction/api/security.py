# api/security.py
import os
import secrets
from typing import Optional
from fastapi import Request
from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBasic, HTTPBasicCredentials

def require_auth(
    request: Request,
    x_api_key: str = Header(default="", alias="X-API-Key"),
    credentials: Optional[HTTPBasicCredentials] = Depends(basic_security),
) -> str:
    """
    Accept either:
    - X-API-Key header (Android/client use)
    - HTTP Basic Auth (browser/manual use)
    """
    # 1) API Key path (preferred for Android)
    if x_api_key and secrets.compare_digest(x_api_key, API_KEY):
        return "api_key"

    # 2) Basic Auth path
    if credentials:
        ok_user = secrets.compare_digest(credentials.username, BASIC_USER)
        ok_pass = secrets.compare_digest(credentials.password, BASIC_PASS)
        if ok_user and ok_pass:
            return credentials.username

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized",
        headers={"WWW-Authenticate": "Basic"},
    )
# -----------------------------
# BASIC AUTH (recommended for Stage 9 LAN)
# -----------------------------
basic_security = HTTPBasic()

BASIC_USER = os.getenv("GARGI_BASIC_USER", "gargi")
BASIC_PASS = os.getenv("GARGI_BASIC_PASS", "sharma")

def require_basic_auth(credentials: HTTPBasicCredentials = Depends(basic_security)) -> str:
    ok_user = secrets.compare_digest(credentials.username, BASIC_USER)
    ok_pass = secrets.compare_digest(credentials.password, BASIC_PASS)

    if not (ok_user and ok_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# -----------------------------
# API KEY (keep for backward compatibility with your current app.py)
# -----------------------------
API_KEY = os.getenv("GARGI_API_KEY", "dev-local-key")

def require_api_key(x_api_key: str = Header(default="", alias="X-API-Key")) -> None:
    """
    Checks X-API-Key header. Keep this for your existing dependencies=[Depends(require_api_key)] usage.
    """
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing/invalid API key",
        )

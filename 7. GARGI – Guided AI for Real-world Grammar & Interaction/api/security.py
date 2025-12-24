# api/security.py
import os
import secrets
import base64
from typing import Optional, Tuple

from fastapi import Depends, HTTPException, status, Header, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials


# -----------------------------
# CONFIG
# -----------------------------
basic_security = HTTPBasic(auto_error=False)

BASIC_USER = os.getenv("GARGI_BASIC_USER", "gargi")
BASIC_PASS = os.getenv("GARGI_BASIC_PASS", "sharma")

API_KEY = os.getenv("GARGI_API_KEY", "dev-local-key")


# -----------------------------
# BASIC AUTH (manual + dependency)
# -----------------------------
def require_basic_auth(credentials: Optional[HTTPBasicCredentials] = Depends(basic_security)) -> str:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Basic"},
        )

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
# API KEY (header)
# -----------------------------
def require_api_key(x_api_key: str = Header(default="", alias="X-API-Key")) -> None:
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing/invalid API key",
        )


# -----------------------------
# ANDROID-READY: accept EITHER API key OR Basic Auth
# -----------------------------
def require_auth(
    x_api_key: str = Header(default="", alias="X-API-Key"),
    credentials: Optional[HTTPBasicCredentials] = Depends(basic_security),
) -> str:
    # 1) API Key path (Android/client)
    if x_api_key and secrets.compare_digest(x_api_key, API_KEY):
        return "api_key"

    # 2) Basic Auth path (browser/manual)
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
# Middleware helper: protect /docs & /openapi.json too
# (FastAPI docs routes do not easily take dependencies)
# -----------------------------
def _parse_basic_auth_header(auth_header: str) -> Optional[Tuple[str, str]]:
    # Expect: "Basic base64(user:pass)"
    try:
        scheme, b64 = auth_header.split(" ", 1)
        if scheme.lower() != "basic":
            return None
        raw = base64.b64decode(b64).decode("utf-8")
        user, pwd = raw.split(":", 1)
        return user, pwd
    except Exception:
        return None


def authorize_request_for_docs(request: Request) -> None:
    """
    Enforces auth for docs/openapi endpoints using either:
    - X-API-Key header, OR
    - Basic Authorization header
    """
    x_api_key = request.headers.get("x-api-key", "")
    if x_api_key and secrets.compare_digest(x_api_key, API_KEY):
        return

    auth = request.headers.get("authorization", "")
    parsed = _parse_basic_auth_header(auth) if auth else None
    if parsed:
        user, pwd = parsed
        if secrets.compare_digest(user, BASIC_USER) and secrets.compare_digest(pwd, BASIC_PASS):
            return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized",
        headers={"WWW-Authenticate": "Basic"},
    )

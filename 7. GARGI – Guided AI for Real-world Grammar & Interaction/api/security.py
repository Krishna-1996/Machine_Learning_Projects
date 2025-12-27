# api/security.py
import os
import secrets
import base64
from typing import Optional, Tuple

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials


# -----------------------------
# CONFIG
# -----------------------------
basic_security = HTTPBasic(auto_error=False)

BASIC_USER = os.getenv("GARGI_BASIC_USER", "gargi")
BASIC_PASS = os.getenv("GARGI_BASIC_PASS", "sharma")
API_KEY = os.getenv("GARGI_API_KEY", "dev-local-key")


def _unauthorized(detail: str = "Unauthorized"):
    # Browser will show login prompt because of WWW-Authenticate: Basic
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Basic"},
    )


# -----------------------------
# BASIC AUTH (dependency)
# -----------------------------
def require_basic_auth(
    credentials: Optional[HTTPBasicCredentials] = Depends(basic_security),
) -> str:
    if credentials is None:
        _unauthorized("Not authenticated")

    ok_user = secrets.compare_digest(credentials.username, BASIC_USER)
    ok_pass = secrets.compare_digest(credentials.password, BASIC_PASS)

    if not (ok_user and ok_pass):
        _unauthorized("Invalid username or password")

    return credentials.username


# -----------------------------
# Helpers
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


def _api_key_matches(provided: Optional[str]) -> bool:
    if not provided:
        return False
    return secrets.compare_digest(provided.strip(), (API_KEY or "").strip())


# -----------------------------
# ANDROID-READY: accept EITHER API key OR Basic Auth
# (Primary = X-API-Key from request.headers)
# -----------------------------
def require_auth(
    request: Request,
    credentials: Optional[HTTPBasicCredentials] = Depends(basic_security),
) -> str:
    """
    Accept either:
    - API Key in header: X-API-Key: <key>   (Android / curl / PowerShell)
    OR
    - Basic Auth: Authorization: Basic ...  (browser)
    """

    # 1) API key path (robust; does not depend on Header injection)
    x_api_key = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
    if _api_key_matches(x_api_key):
        return "api_key"

    # 2) Basic auth path via dependency (works for browser)
    if credentials:
        ok_user = secrets.compare_digest(credentials.username, BASIC_USER)
        ok_pass = secrets.compare_digest(credentials.password, BASIC_PASS)
        if ok_user and ok_pass:
            return "basic"

    _unauthorized("Unauthorized")


# -----------------------------
# Docs protection: BASIC ONLY
# -----------------------------
def authorize_request_for_docs(request: Request) -> str:
    """
    Protect /docs, /redoc, /openapi.json with browser Basic auth.
    (No API key here; keeps docs private.)
    """
    auth = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    parsed = _parse_basic_auth_header(auth)
    if not parsed:
        _unauthorized("Docs require Basic auth")

    user, pwd = parsed
    ok_user = secrets.compare_digest(user, BASIC_USER)
    ok_pass = secrets.compare_digest(pwd, BASIC_PASS)
    if not (ok_user and ok_pass):
        _unauthorized("Invalid username or password")

    return "basic"

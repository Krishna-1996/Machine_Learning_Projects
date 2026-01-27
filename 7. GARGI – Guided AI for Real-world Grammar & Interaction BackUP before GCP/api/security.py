# api/security.py
import os
import secrets
import base64
from typing import Optional, Tuple

from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBasic, HTTPBasicCredentials

basic_security = HTTPBasic(auto_error=False)


# -----------------------------
# Helpers: read ENV at request-time
# (avoids "loaded .env after import" problems)
# -----------------------------
def _get_basic_user() -> str:
    return os.getenv("GARGI_BASIC_USER", "gargi")


def _get_basic_pass() -> str:
    return os.getenv("GARGI_BASIC_PASS", "sharma")


def _get_api_key() -> str:
    # Important: strip to avoid invisible whitespace/newline issues in .env
    return (os.getenv("GARGI_API_KEY", "dev-local-key") or "").strip()


def _unauthorized() -> None:
    # Browser will show login prompt because of WWW-Authenticate: Basic
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized",
        headers={"WWW-Authenticate": "Basic"},
    )


# -----------------------------
# BASIC AUTH dependency
# -----------------------------
def require_basic_auth(
    credentials: Optional[HTTPBasicCredentials] = Depends(basic_security),
) -> str:
    if credentials is None:
        _unauthorized()

    ok_user = secrets.compare_digest(credentials.username, _get_basic_user())
    ok_pass = secrets.compare_digest(credentials.password, _get_basic_pass())

    if not (ok_user and ok_pass):
        _unauthorized()

    return credentials.username


# -----------------------------
# API KEY dependency (header)
# -----------------------------
def require_api_key(
    x_api_key: str = Header(default="", alias="X-API-Key"),
) -> str:
    incoming = (x_api_key or "").strip()
    expected = _get_api_key()

    if not incoming or not expected or not secrets.compare_digest(incoming, expected):
        # Use 401 to be consistent with "auth required" behavior
        _unauthorized()

    return "api_key"


# -----------------------------
# Accept EITHER API key OR Basic Auth
# -----------------------------
def require_auth(
    credentials: Optional[HTTPBasicCredentials] = Depends(basic_security),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> str:
    """
    Accept either:
    - API Key in header: X-API-Key: <key>   (Android / curl)
    OR
    - Basic Auth: Authorization: Basic ...  (browser / manual)
    """

    # 1) API key path
    incoming = (x_api_key or "").strip()
    expected = _get_api_key()
    if incoming and expected and secrets.compare_digest(incoming, expected):
        return "api_key"

    # 2) Basic auth path
    if credentials:
        ok_user = secrets.compare_digest(credentials.username, _get_basic_user())
        ok_pass = secrets.compare_digest(credentials.password, _get_basic_pass())
        if ok_user and ok_pass:
            return "basic"

    _unauthorized()


# -----------------------------
# Optional: Docs protection (Basic only)
# -----------------------------
def authorize_request_for_docs(
    credentials: Optional[HTTPBasicCredentials] = Depends(basic_security),
) -> str:
    if credentials:
        ok_user = secrets.compare_digest(credentials.username, _get_basic_user())
        ok_pass = secrets.compare_digest(credentials.password, _get_basic_pass())
        if ok_user and ok_pass:
            return "basic"
    _unauthorized()


# -----------------------------
# Utility: parse Basic header (if you ever use middleware-style protection)
# -----------------------------
def _parse_basic_auth_header(auth_header: str) -> Optional[Tuple[str, str]]:
    try:
        scheme, b64 = auth_header.split(" ", 1)
        if scheme.lower() != "basic":
            return None
        raw = base64.b64decode(b64).decode("utf-8")
        user, pwd = raw.split(":", 1)
        return user, pwd
    except Exception:
        return None

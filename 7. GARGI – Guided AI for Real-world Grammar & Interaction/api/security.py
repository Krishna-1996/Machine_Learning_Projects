# api/security.py
import os
import secrets
import base64
from typing import Optional, Tuple
from pathlib import Path

from fastapi import Depends, HTTPException, status, Header, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials

# Load .env here as well (robust against import order / reload quirks)
try:
    from dotenv import load_dotenv  # python-dotenv
    ENV_PATH = Path(__file__).resolve().parents[1] / ".env"  # project root/.env
    load_dotenv(ENV_PATH)
except Exception:
    # If dotenv isn't available or load fails, we fall back to os.environ
    pass

basic_security = HTTPBasic(auto_error=False)


def _cfg() -> tuple[str, str, str]:
    """
    Read config at call-time (not import-time) so .env changes apply reliably.
    """
    basic_user = os.getenv("GARGI_BASIC_USER", "gargi")
    basic_pass = os.getenv("GARGI_BASIC_PASS", "sharma")
    api_key = os.getenv("GARGI_API_KEY", "dev-local-key")
    return basic_user, basic_pass, api_key


def _unauthorized(detail: str = "Unauthorized") -> None:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Basic"},
    )


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


def _get_api_key_from_request(request: Request) -> Optional[str]:
    # Starlette headers are case-insensitive; this should work
    v = request.headers.get("x-api-key")
    if v:
        return v
    # Defensive fallbacks
    for k in ("X-API-Key", "X-Api-Key", "x-api-key"):
        v2 = request.headers.get(k)
        if v2:
            return v2
    return None


# -----------------------------
# BASIC AUTH dependency
# -----------------------------
def require_basic_auth(
    credentials: Optional[HTTPBasicCredentials] = Depends(basic_security),
) -> str:
    basic_user, basic_pass, _api_key = _cfg()

    if credentials is None:
        _unauthorized("Not authenticated")

    ok_user = secrets.compare_digest(credentials.username, basic_user)
    ok_pass = secrets.compare_digest(credentials.password, basic_pass)

    if not (ok_user and ok_pass):
        _unauthorized("Invalid username or password")

    return credentials.username


# -----------------------------
# API KEY dependency (optional helper)
# -----------------------------
def require_api_key(
    x_api_key: str = Header(default="", alias="X-API-Key"),
) -> None:
    _basic_user, _basic_pass, api_key = _cfg()
    if not x_api_key or not secrets.compare_digest(x_api_key.strip(), api_key.strip()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing/invalid API key",
        )


# -----------------------------
# Accept EITHER API key OR Basic
# -----------------------------
def require_auth(
    request: Request,
    credentials: Optional[HTTPBasicCredentials] = Depends(basic_security),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> str:
    basic_user, basic_pass, api_key = _cfg()

    # 1) API key path
    key = (x_api_key or "").strip() if x_api_key else ""
    if not key:
        rk = _get_api_key_from_request(request)
        if rk:
            key = rk.strip()

    if key and secrets.compare_digest(key, (api_key or "").strip()):
        return "api_key"

    # 2) Basic auth path
    if credentials:
        ok_user = secrets.compare_digest(credentials.username, basic_user)
        ok_pass = secrets.compare_digest(credentials.password, basic_pass)
        if ok_user and ok_pass:
            return "basic"
    else:
        auth = request.headers.get("authorization") or ""
        parsed = _parse_basic_auth_header(auth) if auth else None
        if parsed:
            user, pwd = parsed
            ok_user = secrets.compare_digest(user, basic_user)
            ok_pass = secrets.compare_digest(pwd, basic_pass)
            if ok_user and ok_pass:
                return "basic"

    _unauthorized()


# -----------------------------
# Docs protection for middleware: Basic ONLY
# -----------------------------
def authorize_request_for_docs(request: Request) -> str:
    basic_user, basic_pass, _api_key = _cfg()

    auth = request.headers.get("authorization") or ""
    parsed = _parse_basic_auth_header(auth) if auth else None
    if not parsed:
        _unauthorized("Not authenticated")

    user, pwd = parsed
    ok_user = secrets.compare_digest(user, basic_user)
    ok_pass = secrets.compare_digest(pwd, basic_pass)

    if not (ok_user and ok_pass):
        _unauthorized("Invalid username or password")

    return "basic"

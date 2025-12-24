# api/security.py
import os
import secrets
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

def require_basic_auth(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    # Use environment variables so you don't hardcode secrets in code.
    user = os.getenv("GARGI_BASIC_USER", "gargi")
    pwd  = os.getenv("GARGI_BASIC_PASS", "change-me-now")

    ok_user = secrets.compare_digest(credentials.username, user)
    ok_pwd  = secrets.compare_digest(credentials.password, pwd)

    if not (ok_user and ok_pwd):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

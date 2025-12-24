import os
from fastapi import Header, HTTPException

def require_api_key(x_api_key: str = Header(default="")) -> None:
    expected = os.getenv("GARGI_API_KEY", "").strip()
    if not expected:
        raise HTTPException(status_code=500, detail="Server missing GARGI_API_KEY.")
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key.")

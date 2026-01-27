from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    """
    Public health endpoint.
    Android uses this to verify backend availability.
    """
    return {
        "status": "ok",
        "service": "gargi-api",
        "version": "0.1"
    }

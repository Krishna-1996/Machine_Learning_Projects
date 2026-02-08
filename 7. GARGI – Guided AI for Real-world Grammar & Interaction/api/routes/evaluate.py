from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class EvaluateRequest(BaseModel):
    transcript: str
    duration_sec: float

class EvaluateResponse(BaseModel):
    duration_sec: float
    word_count: int
    wpm: float

@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    words = len(req.transcript.strip().split())
    minutes = req.duration_sec / 60 if req.duration_sec > 0 else 0
    wpm = round(words / minutes, 2) if minutes > 0 else 0.0

    return EvaluateResponse(
        duration_sec=req.duration_sec,
        word_count=words,
        wpm=wpm
    )

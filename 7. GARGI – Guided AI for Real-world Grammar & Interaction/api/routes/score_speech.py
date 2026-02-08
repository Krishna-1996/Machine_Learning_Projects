from fastapi import APIRouter
from pydantic import BaseModel
from speech_analysis.stage3_analysis import analyze_text_only
from speech_analysis.scoring_engine import score_by_level

router = APIRouter(prefix="/score")

class SpeechScoreRequest(BaseModel):
    level: str
    transcript: str
    duration_sec: float | None = None

@router.post("/speech")
def score_speech(req: SpeechScoreRequest):
    metrics = analyze_text_only(
        transcript=req.transcript,
        duration=req.duration_sec
    )

    result = score_by_level(
        level=req.level,
        metrics=metrics
    )

    return result

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

from scoring.scoring_engine import compute_final_score

app = FastAPI(title="GARGI Backend", version="1.0")


# ---------- Request / Response Schemas ----------

class ScoreRequest(BaseModel):
    level: str
    transcript: str
    analysis: Dict[str, Any]


class ScoreResponse(BaseModel):
    level: str
    total_score: int
    components: Dict[str, float]


# ---------- Routes ----------

@app.get("/")
def health_check():
    return {"status": "GARGI backend running"}


@app.post("/score", response_model=ScoreResponse)
def score_speech(req: ScoreRequest):
    total, components = compute_final_score(
        level=req.level,
        transcript=req.transcript,
        analysis=req.analysis
    )

    return ScoreResponse(
        level=req.level,
        total_score=total,
        components=components
    )

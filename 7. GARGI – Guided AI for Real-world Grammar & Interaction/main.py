from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

from scoring.scoring_engine import compute_final_score
from scoring.weights import LEVEL_WEIGHTS

from feedback.explainability import explain
from feedback.tips import get_tip
from feedback.tone import apply_tone

from progress.storage import save_session, load_sessions
from progress.trends import compute_trends
from progress.bands import score_band
from progress.summary import generate_summary

app = FastAPI(title="GARGI Backend", version="1.2")


class ScoreRequest(BaseModel):
    user_id: str
    level: str
    transcript: str
    analysis: Dict[str, Any]


class ScoreResponse(BaseModel):
    level: str
    total_score: int
    band: str
    components: Dict[str, float]
    explanations: Dict[str, str]
    tips: Dict[str, str]
    trends: Dict[str, float]
    summary: str


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    total, components = compute_final_score(
        level=req.level,
        transcript=req.transcript,
        analysis=req.analysis
    )

    explanations = {}
    tips = {}
    weights = LEVEL_WEIGHTS[req.level]

    for metric, weighted_score in components.items():
        raw_estimate = int(weighted_score / weights[metric])
        explanations[metric] = apply_tone(
            explain(metric, raw_estimate), req.level
        )
        tips[metric] = get_tip(metric)

    session_payload = {
        "total_score": total,
        "components": components
    }

    save_session(req.user_id, session_payload)
    sessions = load_sessions(req.user_id)

    trends = compute_trends(sessions)
    band = score_band(total)
    summary = generate_summary(req.level, total, trends)

    return ScoreResponse(
        level=req.level,
        total_score=total,
        band=band,
        components=components,
        explanations=explanations,
        tips=tips,
        trends=trends,
        summary=summary
    )

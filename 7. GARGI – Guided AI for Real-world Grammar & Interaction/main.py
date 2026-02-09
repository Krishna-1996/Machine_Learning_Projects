from firebase.firebase_init import init_firebase
from firebase.firestore_client import save_session

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

from dashboard.analytics import (
    load_all_users,
    learner_overview,
    learner_detail,
    class_overview
)
from dashboard.risk_flags import compute_risk_flags
from dashboard.schemas import (
    LearnerOverview,
    LearnerDetail,
    ClassOverview
)

from auth.schemas import LoginRequest, TokenResponse
from auth.auth import authenticate, create_access_token

from dashboard.views import register_dashboard_routes

@app.on_event("startup")
def startup():
    init_firebase()

# Initialize FastAPI app
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

    response = ScoreResponse(
    level=req.level,
    total_score=total,
    band=band,
    components=components,
    explanations=explanations,
    tips=tips,
    trends=trends,
    summary=summary
)

return response



@app.get("/dashboard/learners", response_model=list[LearnerOverview])
def get_all_learners():
    users = load_all_users()
    return [
        learner_overview(uid, sessions)
        for uid, sessions in users.items()
    ]


@app.get("/dashboard/learners/{user_id}", response_model=LearnerDetail)
def get_learner_detail(user_id: str):
    users = load_all_users()
    sessions = users.get(user_id)

    if not sessions:
        return {
            "user_id": user_id,
            "scores": [],
            "components_history": {},
            "trends": {},
            "risk_flags": ["No data available"]
        }

    scores, components, trends = learner_detail(user_id, sessions)
    flags = compute_risk_flags(scores, trends)

    return {
        "user_id": user_id,
        "scores": scores,
        "components_history": components,
        "trends": trends,
        "risk_flags": flags
    }


@app.get("/dashboard/class", response_model=ClassOverview)
def get_class_overview():
    users = load_all_users()
    overview = class_overview(users)

    at_risk = []
    for uid, sessions in users.items():
        scores, _, trends = learner_detail(uid, sessions)
        if compute_risk_flags(scores, trends):
            at_risk.append(uid)

    overview["at_risk_learners"] = at_risk
    return overview

@app.post("/auth/login", response_model=TokenResponse)
def login(data: LoginRequest):
    user = authenticate(data.username, data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(user)
    return {"access_token": token}

register_dashboard_routes(app)

@app.get("/health")
def health():
    return {"status": "ok"}

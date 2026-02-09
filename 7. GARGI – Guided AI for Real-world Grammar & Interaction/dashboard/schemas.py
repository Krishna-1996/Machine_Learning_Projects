from pydantic import BaseModel
from typing import Dict, List


class LearnerOverview(BaseModel):
    user_id: str
    attempts: int
    latest_score: int
    band: str
    avg_score: float


class LearnerDetail(BaseModel):
    user_id: str
    scores: List[int]
    components_history: Dict[str, List[float]]
    trends: Dict[str, float]
    risk_flags: List[str]


class ClassOverview(BaseModel):
    total_learners: int
    average_score: float
    band_distribution: Dict[str, int]
    at_risk_learners: List[str]

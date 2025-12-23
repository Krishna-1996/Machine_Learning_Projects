from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class TopicObj(BaseModel):
    # flexible schema: allow any extra fields from your enriched CSV
    topic_id: Optional[int] = None
    category: Optional[str] = None
    topic_raw: Optional[str] = None
    instruction: Optional[str] = None
    topic_content: Optional[str] = None
    topic_type: Optional[str] = None
    constraints: Optional[str] = None
    expected_anchors: Optional[List[str]] = None
    topic_keyphrases: Optional[List[str]] = None

    class Config:
        extra = "allow"


class TopicResponse(BaseModel):
    topic_obj: Dict[str, Any]
    topic_text: str


class EvaluateTextRequest(BaseModel):
    transcript: str = Field(..., min_length=3)

    # Either provide topic_obj or topic_text
    topic_obj: Optional[Dict[str, Any]] = None
    topic_text: Optional[str] = None

    # Optional: if client knows duration
    duration_sec: Optional[float] = None


class EvaluateTextResponse(BaseModel):
    topic_obj: Dict[str, Any]
    transcript: str

    stage3: Dict[str, Any]
    stage4: Dict[str, Any]
    stage5: Dict[str, Any]
    stage6: Dict[str, Any]

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class ApiMeta(BaseModel):
    request_id: str
    elapsed_ms: Optional[int] = None


class ApiError(BaseModel):
    code: str
    message: str


class ApiResponse(BaseModel):
    ok: bool = True
    meta: ApiMeta
    error: Optional[ApiError] = None
    data: Optional[Dict[str, Any]] = None


class TopicResponse(BaseModel):
    topic_obj: Dict[str, Any]
    topic_text: str


class EvaluateTextRequest(BaseModel):
    transcript: str = Field(..., min_length=3)
    topic_obj: Optional[Dict[str, Any]] = None
    topic_text: Optional[str] = None
    duration_sec: Optional[float] = None
    save_history: bool = True
    user_id: Optional[str] = None


class EvaluateTextResponse(BaseModel):
    topic_obj: Dict[str, Any]
    topic_text: str
    transcript: str
    stage3: Dict[str, Any]
    stage4: Dict[str, Any]
    stage5: Dict[str, Any]
    stage6: Dict[str, Any]
    user_id: Optional[str] = None

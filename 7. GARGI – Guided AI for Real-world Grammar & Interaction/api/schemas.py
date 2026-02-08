from __future__ import annotations

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


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
    topic_obj: Dict[str, Any] = Field(default_factory=dict)
    topic_text: str = ""


class EvaluateTextRequest(BaseModel):
    transcript: str = Field(..., min_length=1)
    duration_sec: Optional[float] = Field(default=None, ge=0)
    topic_obj: Optional[Dict[str, Any]] = None
    topic_text: Optional[str] = None
    duration_sec: Optional[float] = None
    save_history: bool = True
    user_id: Optional[str] = None


class EvaluateTextResponse(BaseModel):
    topic_obj: Dict[str, Any] = Field(default_factory=dict)
    topic_text: str = ""
    transcript: str = ""
    stage3: Dict[str, Any] = Field(default_factory=dict)
    stage4: Dict[str, Any] = Field(default_factory=dict)
    stage5: Dict[str, Any] = Field(default_factory=dict)
    stage6: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
class EvaluateEnvelope(BaseModel):
    # Keep it flexible if your pipeline changes
    topic: Optional[str] = None
    transcript: str
    duration_sec: Optional[float] = None
    meta: Dict[str, Any] = {}
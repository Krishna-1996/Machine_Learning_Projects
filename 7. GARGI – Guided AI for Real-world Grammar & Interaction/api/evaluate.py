# D:\Machine_Learning_Projects\7. GARGI – Guided AI for Real-world Grammar & Interaction\api\evaluate.py

from __future__ import annotations

import traceback
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field


router = APIRouter(tags=["evaluate"])


# -----------------------------
# Request/Response models
# -----------------------------
class EvaluateTextRequest(BaseModel):
    """
    Request payload for evaluating a text transcript.

    Notes:
    - user_id and save_history are accepted for future use, but MUST NOT be passed into
      run_stage6() unless run_stage6 supports those arguments.
    """

    transcript: str = Field(..., min_length=1)
    topic_text: Optional[str] = None
    topic_obj: Optional[Dict[str, Any]] = None
    duration_sec: Optional[float] = None

    # accepted but not necessarily used by Stage 6
    user_id: Optional[str] = None
    save_history: Optional[bool] = None


class EvaluateTextResponse(BaseModel):
    """
    Minimal response that matches your Android DTO:
        data class EvaluateTextResponseDto(val result: String)
    """
    result: str


# -----------------------------
# Internal helpers
# -----------------------------
def _get_request_id(incoming: Optional[str]) -> str:
    return incoming.strip() if incoming and incoming.strip() else str(uuid.uuid4())


def _import_run_stage6():
    """
    Lazy import to avoid circular imports / env loading timing problems.

    IMPORTANT:
    Adjust the import paths below to match your project structure if needed.
    We try a few common paths to reduce friction.
    """
    candidates = [
        # Example candidates — adjust if your actual stage6 module differs
        ("core.stage6", "run_stage6"),
        ("core.pipeline.stage6", "run_stage6"),
        ("gargi_assistant.core.stage6", "run_stage6"),
        ("api.pipeline.stage6", "run_stage6"),
        ("stage6", "run_stage6"),
    ]

    last_err: Optional[Exception] = None
    for module_name, fn_name in candidates:
        try:
            mod = __import__(module_name, fromlist=[fn_name])
            fn = getattr(mod, fn_name)
            return fn
        except Exception as e:
            last_err = e

    raise ImportError(
        "Could not import run_stage6(). "
        "Update _import_run_stage6() candidates to the correct module path."
    ) from last_err


def _coerce_result_to_string(stage6_out: Any) -> str:
    """
    We normalize whatever Stage 6 returns into a single string:
    - If Stage6 returns dict, try common keys.
    - If it returns string, use it directly.
    - Otherwise, str().
    """
    if stage6_out is None:
        return "No evaluation result produced."

    if isinstance(stage6_out, str):
        return stage6_out

    if isinstance(stage6_out, dict):
        # Try common keys you might already use in your pipeline
        for key in ("result", "final_feedback", "feedback", "summary", "text"):
            val = stage6_out.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        # fallback: compact dict
        return str(stage6_out)

    return str(stage6_out)


# -----------------------------
# Route
# -----------------------------
@router.post(
    "/evaluate/text",
    response_model=EvaluateTextResponse,
)
def evaluate_text(
    req: EvaluateTextRequest,
    # This allows you to attach your own request id from client, else server generates one.
    x_request_id: Optional[str] = Header(default=None, alias="X-Request-Id"),
    # Auth dependency (API Key OR Basic). Imported here to avoid import timing issues.
    _auth: str = Depends(lambda: __import__("api.security", fromlist=["require_auth"]).require_auth),
):
    request_id = _get_request_id(x_request_id)

    # Basic validation
    transcript = (req.transcript or "").strip()
    if not transcript:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="transcript is required",
            headers={"X-Request-Id": request_id},
        )

    # IMPORTANT: sanitize arguments sent to run_stage6:
    # Do NOT forward user_id/save_history unless Stage6 supports them.
    safe_kwargs = {
        "transcript": transcript,
        "topic_text": req.topic_text,
        "topic_obj": req.topic_obj,
        "duration_sec": req.duration_sec,
    }

    # Remove None keys to keep stage6 cleaner
    safe_kwargs = {k: v for k, v in safe_kwargs.items() if v is not None}

    try:
        run_stage6 = _import_run_stage6()

        # If your run_stage6 expects different arg names, adjust here.
        stage6_out = run_stage6(**safe_kwargs)

        result_text = _coerce_result_to_string(stage6_out)

        return EvaluateTextResponse(result=result_text)

    except TypeError as e:
        # This catches "unexpected keyword argument" cleanly, with guidance.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"TypeError in Stage6 call: {e}. "
                f"Stage6 kwargs sent: {list(safe_kwargs.keys())}. "
                "Fix run_stage6 signature OR adjust safe_kwargs mapping."
            ),
            headers={"X-Request-Id": request_id},
        )

    except Exception:
        # Don't leak stack traces to clients by default.
        # If you want debug mode later, we can add a DEBUG flag.
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
            headers={"X-Request-Id": request_id},
        )

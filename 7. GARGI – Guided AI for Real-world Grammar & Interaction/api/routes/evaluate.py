from __future__ import annotations

import time
import uuid
from typing import Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Ensure project root imports resolve (same pattern as topics.py)
import api.deps  # noqa: F401

from speech_analysis.stage3_text_analysis import run_stage3_text
from speech_analysis.stage4_scoring import run_stage4
from topic_relevance.stage5_relevance import run_stage5
from speech_analysis.stage6_feedback import run_stage6
from api.deps import EvaluateEnvelope, _format_multiline_feedback


router = APIRouter(prefix="", tags=["evaluate"])


class EvaluateTextRequest(BaseModel):
    transcript: str
    duration_sec: int
    topic_text: str = ""
    topic_obj: Dict[str, Any] | None = None
    save_history: bool = True
    user_id: str | None = None


@router.post("/evaluate/text", response_model=EvaluateEnvelope)
def evaluate_text(req: EvaluateTextRequest):
    start = time.time()
    request_id = str(uuid.uuid4())

    transcript = (req.transcript or "").strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript is empty")

    topic_text = (req.topic_text or "").strip()
    topic_obj = req.topic_obj or {"topic_raw": topic_text}

    try:
        # ---------------- Stage 3 (TEXT) ----------------
        stage3 = run_stage3_text(
            transcript=transcript,
            duration_sec=req.duration_sec,
        )

        # ---------------- Stage 4 -----------------------
        stage4 = run_stage4(stage3)

        # ---------------- Stage 5 -----------------------
        stage5 = run_stage5(topic_obj, transcript)

        # ---------------- Stage 6 -----------------------
        stage6 = run_stage6(
            topic_text=topic_text,
            transcript=transcript,
            stage4_results=stage4,
            stage5_results=stage5,
            save_history=bool(req.save_history),
        )

        result_text = _format_multiline_feedback(
            stage3, stage4, stage5, stage6
        )

        raw = {
            "topic_obj": topic_obj,
            "topic_text": topic_text,
            "transcript": transcript,
            "stage3": stage3,
            "stage4": stage4,
            "stage5": stage5,
            "stage6": stage6,
            "meta": {
                "request_id": request_id,
                "elapsed_ms": int((time.time() - start) * 1000),
                "save_history": bool(req.save_history),
                "user_id": req.user_id,
            },
        }

        return EvaluateEnvelope(
            ok=True,
            request_id=request_id,
            result=result_text,
            raw=raw,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {e}",
        )

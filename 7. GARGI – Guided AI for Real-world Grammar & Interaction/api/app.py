# D:\Machine_Learning_Projects\7. GARGI – Guided AI for Real-world Grammar & Interaction\api\routes\evaluate.py
from __future__ import annotations

import time
import uuid
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.schemas import EvaluateTextRequest

# ✅ Correct module locations (VERY IMPORTANT)
from scoring_feedback.stage4_scoring import run_stage4
from topic_relevance.stage5_relevance import run_stage5
from coaching.stage6_coaching import run_stage6

# ✅ Text-only Stage 3 (new file)
from speech_analysis.stage3_text_analysis import run_stage3_text

router = APIRouter(prefix="", tags=["evaluate"])


class EvaluateEnvelope(BaseModel):
    ok: bool
    request_id: str
    result: str
    raw: Dict[str, Any]


def _mk_request_id() -> str:
    return str(uuid.uuid4())


def _topic_text_from_req(req: EvaluateTextRequest) -> str:
    # Prefer topic_text if passed
    if getattr(req, "topic_text", None) and (req.topic_text or "").strip():
        return req.topic_text.strip()

    # Else try topic_obj
    if getattr(req, "topic_obj", None) and isinstance(req.topic_obj, dict):
        t = (req.topic_obj.get("topic_raw") or req.topic_obj.get("topic") or "").strip()
        if t:
            return t

    return "General speaking practice (no topic provided)"


def _format_multiline_feedback(stage3: dict, stage4: dict, stage5: dict, stage6: dict) -> str:
    scores = (stage4.get("scores") or {}) if isinstance(stage4, dict) else {}
    overall = scores.get("overall", "N/A")
    fluency = scores.get("fluency", "N/A")
    grammar = scores.get("grammar", "N/A")
    fillers = scores.get("fillers", "N/A")

    flu = (stage3.get("fluency") or {}) if isinstance(stage3, dict) else {}
    wpm = flu.get("wpm", "N/A")
    duration_sec = flu.get("duration_sec", "N/A")

    relevance_score = stage5.get("relevance_score", "N/A")
    relevance_label = stage5.get("label", "N/A")
    on_topic_ratio = stage5.get("on_topic_sentence_ratio", "N/A")

    conf = (stage6.get("confidence") or {}) if isinstance(stage6, dict) else {}
    conf_score = conf.get("confidence_score", "N/A")
    conf_label = conf.get("confidence_label", "N/A")

    priorities = stage6.get("priorities") or []
    coaching = stage6.get("coaching_feedback") or []
    reflection = stage6.get("reflection_prompts") or []

    lines = []
    lines.append("GARGI Feedback (Text Evaluation)")
    lines.append("-" * 28)
    lines.append(f"Duration: {duration_sec}s | WPM: {wpm}")
    lines.append(f"Scores (0-10): Fluency={fluency} | Grammar={grammar} | Fillers={fillers}")
    lines.append(f"Overall: {overall}")
    lines.append(f"Relevance: {relevance_score} ({relevance_label})")
    lines.append(f"On-topic sentence ratio: {on_topic_ratio}")
    lines.append("")
    lines.append(f"Confidence: {conf_score} ({conf_label})")

    why = conf.get("why")
    if why:
        lines.append(f"Why: {why}")

    if priorities:
        lines.append("")
        lines.append("Top priorities for your next attempt:")
        for i, p in enumerate(priorities[:3], start=1):
            area = p.get("area", "Priority")
            severity = p.get("severity", "Medium")
            reason = p.get("reason", "")
            action = p.get("action", "")
            lines.append(f"{i}) {area} [{severity}]")
            if reason:
                lines.append(f"   Reason: {reason}")
            if action:
                lines.append(f"   Action: {action}")

    if coaching:
        lines.append("")
        lines.append("Coaching feedback:")
        for c in coaching[:6]:
            lines.append(f"- {c}")

    if reflection:
        lines.append("")
        lines.append("Quick reflection prompts:")
        for r in reflection[:6]:
            lines.append(f"- {r}")

    return "\n".join(lines).strip()


@router.post("/evaluate/text", response_model=EvaluateEnvelope)
def evaluate_text(req: EvaluateTextRequest) -> EvaluateEnvelope:
    start = time.time()
    request_id = _mk_request_id()

    transcript = (req.transcript or "").strip()
    if len(transcript) < 3:
        raise HTTPException(status_code=422, detail="Transcript too short.")

    topic_text = _topic_text_from_req(req)

    # Ensure topic_obj exists for Stage 5
    topic_obj = req.topic_obj if isinstance(getattr(req, "topic_obj", None), dict) else {"topic_raw": topic_text}

    try:
        # ✅ Stage 3 from transcript + duration
        stage3 = run_stage3_text(transcript=transcript, duration_sec=req.duration_sec)

        # ✅ Stage 4 now receives correct stage3 schema
        stage4 = run_stage4(stage3)

        # ✅ Stage 5 relevance
        stage5 = run_stage5(topic_obj, transcript)

        # ✅ Stage 6 coaching (also does history save if your implementation does)
        stage6 = run_stage6(
            topic_text=topic_text,
            transcript=transcript,
            stage4_results=stage4,
            stage5_results=stage5,
            save_history=bool(getattr(req, "save_history", False)),
        )

        result_text = _format_multiline_feedback(stage3, stage4, stage5, stage6)

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
                "save_history": bool(getattr(req, "save_history", False)),
                "user_id": getattr(req, "user_id", None),
            },
        }

        return EvaluateEnvelope(ok=True, request_id=request_id, result=result_text, raw=raw)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")

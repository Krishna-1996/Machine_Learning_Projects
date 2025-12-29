from __future__ import annotations

import traceback
from typing import Any, Dict, Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import api.deps  # noqa: F401

from topic_generation.generate_topic import get_random_topic
from speech_analysis.stage3_analysis import analyze_fillers, analyze_grammar
from scoring_feedback.stage4_scoring import run_stage4
from topic_relevance.stage5_relevance import run_stage5
from coaching.stage6_coaching import run_stage6

router = APIRouter(tags=["evaluate"])


# -----------------------------
# Request/Response models
# -----------------------------
class EvaluateTextRequest(BaseModel):
    transcript: str = Field(..., min_length=1)
    topic_text: Optional[str] = None
    topic_obj: Optional[Dict[str, Any]] = None

    duration_sec: int = Field(default=60, ge=5, le=600)
    save_history: bool = Field(default=False)

    user_id: Optional[str] = None


class EvaluateTextResponse(BaseModel):
    result: str
    raw: Dict[str, Any]


# -----------------------------
# Helpers
# -----------------------------
def _compute_wpm(transcript: str, duration_sec: int) -> float:
    words = [w for w in (transcript or "").strip().split() if w.strip()]
    minutes = max(duration_sec, 1) / 60.0
    return round(len(words) / minutes, 2)


def _safe_count_grammar_errors(grammar_out: Any) -> int:
    if grammar_out is None:
        return 0
    if isinstance(grammar_out, list):
        return len(grammar_out)
    if isinstance(grammar_out, dict):
        matches = grammar_out.get("matches")
        if isinstance(matches, list):
            return len(matches)
    return 0


def _build_min_topic_obj(topic_text: str) -> Dict[str, Any]:
    tt = (topic_text or "").strip()
    return {
        "topic_id": None,
        "category": None,
        "topic_raw": tt,
        "instruction": None,
        "topic_content": tt,
        "topic_type": None,
        "constraints": [],
        "expected_anchors": [],
        "topic_keyphrases": [],
    }


def _extract_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for k in ("final", "score", "value"):
            v = value.get(k)
            if isinstance(v, (int, float)):
                return float(v)
    return None


def _normalize_text(s: str) -> str:
    """
    Avoid odd dash/encoding artifacts on some clients.
    Replace common 'smart' characters and typical UTF-8 mojibake sequences.
    """
    if not s:
        return s

    # Normalize common smart punctuation
    s = (
        s.replace("–", "-")
         .replace("—", "-")
         .replace("’", "'")
         .replace("“", '"')
         .replace("”", '"')
    )

    # Fix common mojibake patterns (UTF-8 decoded as Windows-1252)
    s = (
        s.replace("â€“", "-")
         .replace("â€”", "-")
         .replace("â€™", "'")
         .replace("â€œ", '"')
         .replace("â€�", '"')
         .replace("â†’", "->")   # arrow
    )

    # Fix multiplication sign mojibake
    s = s.replace("Ã—", "x")

    # Extra: normalize any remaining strange sequences that appear in your logs
    s = s.replace("Ã", "").replace("â", "")

    return s


def _format_result(stage4: Dict[str, Any], stage5: Dict[str, Any], stage6: Dict[str, Any]) -> str:
    conf = stage6.get("confidence", {}) or {}
    priorities: List[Dict[str, Any]] = stage6.get("priorities", []) or []
    coaching_feedback: List[str] = stage6.get("coaching_feedback", []) or []
    reflection: List[str] = stage6.get("reflection_prompts", []) or []

    # NOTE: overall score lives under stage4.scores.overall in your pipeline
    scores = (stage4.get("scores") or {})
    overall = _extract_score(scores.get("overall"))

    rel_score = stage5.get("relevance_score", None)
    rel_label = stage5.get("label", None)
    on_topic = stage5.get("on_topic_sentence_ratio", None)

    flu = _extract_score(scores.get("fluency"))
    fill = _extract_score(scores.get("fillers"))
    gram = _extract_score(scores.get("grammar"))

    lines: List[str] = []
    lines.append("GARGI Feedback (Text Evaluation)")
    lines.append("--------------------------------")

    if overall is not None:
        overall_s = str(round(overall, 2)).rstrip("0").rstrip(".")
        lines.append(f"Overall Quality Score: {overall_s}")

    if flu is not None or gram is not None or fill is not None:
        flu_s = "N/A" if flu is None else str(round(flu, 2)).rstrip("0").rstrip(".")
        gram_s = "N/A" if gram is None else str(round(gram, 2)).rstrip("0").rstrip(".")
        fill_s = "N/A" if fill is None else str(round(fill, 2)).rstrip("0").rstrip(".")
        lines.append(f"Scores (0-10): Fluency={flu_s} | Grammar={gram_s} | Fillers={fill_s}")

    if rel_score is not None or rel_label is not None:
        lines.append(f"Relevance: {rel_score} ({rel_label})")
    if on_topic is not None:
        lines.append(f"On-topic sentence ratio: {on_topic}")

    c_score = conf.get("confidence_score", None)
    c_label = conf.get("confidence_label", None)
    c_expl = conf.get("confidence_explanation", None)
    if c_score is not None or c_label is not None:
        lines.append("")
        lines.append(f"Confidence: {c_score} ({c_label})")
        if c_expl:
            lines.append(f"Why: {c_expl}")

    if priorities:
        lines.append("")
        lines.append("Top priorities for your next attempt:")
        for i, p in enumerate(priorities[:3], start=1):
            area = p.get("area", "Unknown")
            sev = p.get("severity", "N/A")
            reason = p.get("reason", "")
            action = p.get("action", "")
            lines.append(f"{i}) {area} [{sev}]")
            if reason:
                lines.append(f"   Reason: {reason}")
            if action:
                lines.append(f"   Action: {action}")

    if coaching_feedback:
        lines.append("")
        lines.append("Coaching feedback:")
        for s in coaching_feedback[:8]:
            lines.append(f"- {s}")

    if reflection:
        lines.append("")
        lines.append("Quick reflection prompts:")
        for q in reflection[:4]:
            lines.append(f"- {q}")

    return _normalize_text("\n".join(lines).strip())


def _force_wpm_and_pause(stage3: Dict[str, Any], stage4: Dict[str, Any], stage6: Dict[str, Any]) -> None:
    """
    Ensure a single source of truth:
      - stage3 holds telemetry
      - stage4 evidence reflects stage3
      - stage6 session_summary reflects stage3
    """
    wpm = float(stage3.get("wpm") or 0.0)
    pause_ratio = float(stage3.get("pause_ratio") or 0.0)

    evidence = stage4.get("evidence")
    if isinstance(evidence, dict):
        evidence["wpm"] = wpm
        evidence["pause_ratio"] = pause_ratio

    session_summary = stage6.get("session_summary")
    if isinstance(session_summary, dict):
        session_summary["wpm"] = wpm
        session_summary["pause_ratio"] = pause_ratio


# -----------------------------
# Routes
# -----------------------------
@router.post("/evaluate/text", response_model=EvaluateTextResponse)
def evaluate_text(req: EvaluateTextRequest) -> EvaluateTextResponse:
    try:
        transcript = (req.transcript or "").strip()
        if not transcript:
            raise HTTPException(status_code=422, detail="transcript must not be empty")

        topic_obj = req.topic_obj
        topic_text = (req.topic_text or "").strip()

        if not topic_obj:
            if topic_text:
                topic_obj = _build_min_topic_obj(topic_text)
            else:
                topic_obj = get_random_topic(category=None)
                topic_text = (topic_obj.get("topic_raw") or "").strip()

        if not topic_text:
            topic_text = (topic_obj.get("topic_raw") or "").strip()

        # --- Stage 3 ---
        wpm = _compute_wpm(transcript, req.duration_sec)
        filler_words = analyze_fillers(transcript)

        try:
            grammar_out = analyze_grammar(transcript)
        except Exception:
            grammar_out = []
        grammar_errors_count = _safe_count_grammar_errors(grammar_out)

        stage3_results = {
            "wpm": wpm,
            "pause_ratio": 0.0,  # still placeholder until you compute pause ratio
            "filler_words": filler_words,
            "grammar_errors": grammar_errors_count,
            "grammar_raw": grammar_out,
            "transcript": transcript,
            "duration_sec": req.duration_sec,
        }

        # --- Stage 4 ---
        stage4_results = run_stage4(stage3_results)

        # --- Stage 5 ---
        stage5_results = run_stage5(topic_obj, transcript)

        # --- Stage 6 ---
        stage6_results = run_stage6(
            topic_text,
            transcript,
            stage4_results,
            stage5_results,
            save_history=req.save_history,
        )

        # Normalize telemetry across stages
        _force_wpm_and_pause(stage3_results, stage4_results, stage6_results)

        result_text = _format_result(stage4_results, stage5_results, stage6_results)

        raw = {
            "topic_text": topic_text,
            "topic_obj": topic_obj,
            "stage3": stage3_results,
            "stage4": stage4_results,
            "stage5": stage5_results,
            "stage6": stage6_results,
        }

        return EvaluateTextResponse(result=result_text, raw=raw)

    except HTTPException:
        raise
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

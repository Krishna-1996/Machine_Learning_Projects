from api.deps import PROJECT_ROOT  # noqa: F401

from fastapi import FastAPI, HTTPException
from typing import Optional, Dict, Any

from api.schemas import TopicResponse, EvaluateTextRequest, EvaluateTextResponse

from topic_generation.generate_topic import get_random_topic

from speech_analysis.stage3_analysis import analyze_fillers, calculate_wpm, analyze_grammar
from scoring_feedback.stage4_scoring import run_stage4
from topic_relevance.stage5_relevance import run_stage5

# NOTE: You must have this in your project.
# If your Stage 6 function name differs, paste it and I will align it exactly.
from coaching.stage6_coaching import run_stage6


app = FastAPI(
    title="GARGI API",
    version="8.1",
    description="Guided AI for Real-world General Interaction — FastAPI Layer"
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "gargi-api", "version": "8.1"}


def _normalize_topic_obj(topic_obj: Dict[str, Any] | None, topic_text: str | None) -> Dict[str, Any]:
    """
    Ensure Stage 5 receives a metadata-aware topic_obj with required keys.
    """
    if topic_obj and isinstance(topic_obj, dict):
        # Ensure at least topic_raw exists
        if not topic_obj.get("topic_raw"):
            # some generators use "topic"
            t = topic_obj.get("topic") or topic_obj.get("topic_text") or topic_obj.get("topic_content")
            if t:
                topic_obj["topic_raw"] = t
        if not topic_obj.get("topic_content"):
            topic_obj["topic_content"] = topic_obj.get("topic_raw", "")
        if not topic_obj.get("topic_type"):
            topic_obj["topic_type"] = "general"
        if topic_obj.get("expected_anchors") is None:
            topic_obj["expected_anchors"] = []
        if topic_obj.get("topic_keyphrases") is None:
            topic_obj["topic_keyphrases"] = []
        return topic_obj

    if topic_text and isinstance(topic_text, str) and topic_text.strip():
        t = topic_text.strip()
        return {
            "topic_raw": t,
            "topic_content": t,
            "topic_type": "general",
            "expected_anchors": [],
            "topic_keyphrases": []
        }

    raise HTTPException(status_code=400, detail="Provide either topic_obj or topic_text.")


@app.get("/topics", response_model=TopicResponse)
def topics(category: Optional[str] = None):
    """
    Get one random topic (enriched if your generator returns enriched rows).
    """
    try:
        topic_obj = get_random_topic(category=category)
        if not isinstance(topic_obj, dict):
            raise ValueError("Topic generator returned non-dict.")

        # normalize for downstream Stage 5
        topic_obj = _normalize_topic_obj(topic_obj=topic_obj, topic_text=None)

        topic_text = topic_obj.get("topic_raw") or topic_obj.get("topic_content")
        if not topic_text:
            raise ValueError("Topic text not found in topic_obj.")

        return TopicResponse(topic_obj=topic_obj, topic_text=topic_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Topic generation failed: {e}")


@app.post("/evaluate/text", response_model=EvaluateTextResponse)
def evaluate_text(payload: EvaluateTextRequest):
    """
    Evaluate transcript text vs topic.
    This is the core endpoint for Android later.
    """
    transcript = (payload.transcript or "").strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript is empty.")

    topic_obj = _normalize_topic_obj(payload.topic_obj, payload.topic_text)

    # -----------------------------
    # Stage 3 (text-only subset)
    # -----------------------------
    duration_sec = float(payload.duration_sec) if payload.duration_sec else 60.0
    wpm = calculate_wpm(transcript, duration_sec)
    filler_words = analyze_fillers(transcript)
    grammar = analyze_grammar(transcript)

    stage3 = {
        "fluency": {
            "duration_sec": round(duration_sec, 2),
            "wpm": wpm,
            "pause_ratio": 0.0,  # Not computed in text-only mode
            "filler_words": filler_words
        },
        "grammar": grammar
    }

    # -----------------------------
    # Stage 4
    # -----------------------------
    stage4 = run_stage4(stage3)

    # -----------------------------
    # Stage 5
    # -----------------------------
    stage5 = run_stage5(topic_obj=topic_obj, transcript=transcript)

    # -----------------------------
    # Stage 6 (coaching + logging)
    # -----------------------------
    # Your Stage 6 should save sessions.jsonl and return guidance + confidence
    stage6 = run_stage6(
        topic_obj=topic_obj,
        transcript=transcript,
        stage3=stage3,
        stage4=stage4,
        stage5=stage5,
        save=True
    )

    return EvaluateTextResponse(
        topic_obj=topic_obj,
        transcript=transcript,
        stage3=stage3,
        stage4=stage4,
        stage5=stage5,
        stage6=stage6
    )

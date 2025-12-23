from api.deps import PROJECT_ROOT  # noqa: F401

from fastapi import FastAPI, HTTPException
from typing import Optional, Dict, Any

from api.schemas import TopicResponse, EvaluateTextRequest, EvaluateTextResponse

from topic_generation.generate_topic import get_random_topic

from speech_analysis.stage3_analysis import analyze_fillers, calculate_wpm, analyze_grammar
from scoring_feedback.stage4_scoring import run_stage4
from topic_relevance.stage5_relevance import run_stage5
from coaching.stage6_coaching import run_stage6


app = FastAPI(
    title="GARGI API",
    version="8.1",
    description="Guided AI for Real-world General Interaction — FastAPI Layer"
)

@app.get("/")
def root():
    return {
        "service": "gargi-api",
        "status": "ok",
        "health": "/health",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }

@app.get("/health")
def health():
    return {"status": "ok", "service": "gargi-api", "version": "8.1"}


def normalize_topic_obj(topic_obj: Optional[Dict[str, Any]], topic_text: Optional[str]) -> Dict[str, Any]:
    if topic_obj and isinstance(topic_obj, dict):
        # Ensure required keys exist for Stage 5
        if not topic_obj.get("topic_raw"):
            t = topic_obj.get("topic") or topic_obj.get("topic_content") or topic_obj.get("topic_text")
            if t:
                topic_obj["topic_raw"] = t

        if not topic_obj.get("topic_content"):
            topic_obj["topic_content"] = topic_obj.get("topic_raw", "")

        topic_obj.setdefault("topic_type", "general")
        topic_obj.setdefault("expected_anchors", [])
        topic_obj.setdefault("topic_keyphrases", [])
        return topic_obj

    if topic_text and topic_text.strip():
        t = topic_text.strip()
        return {
            "topic_raw": t,
            "topic_content": t,
            "topic_type": "general",
            "expected_anchors": [],
            "topic_keyphrases": [],
        }

    raise HTTPException(status_code=400, detail="Provide either topic_obj or topic_text.")


def topic_text_from_obj(topic_obj: Dict[str, Any]) -> str:
    return (topic_obj.get("topic_raw") or topic_obj.get("topic_content") or "").strip()


@app.get("/topics", response_model=TopicResponse)
def topics(category: Optional[str] = None):
    try:
        topic_obj = get_random_topic(category=category)
        if not isinstance(topic_obj, dict):
            raise ValueError("Topic generator returned non-dict.")

        topic_obj = normalize_topic_obj(topic_obj, None)
        t = topic_text_from_obj(topic_obj)
        if not t:
            raise ValueError("Topic text missing in topic_obj.")

        return TopicResponse(topic_obj=topic_obj, topic_text=t)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Topic generation failed: {e}")


@app.post("/evaluate/text", response_model=EvaluateTextResponse)
def evaluate_text(payload: EvaluateTextRequest):
    transcript = (payload.transcript or "").strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript is empty.")

    topic_obj = normalize_topic_obj(payload.topic_obj, payload.topic_text)
    topic_text = topic_text_from_obj(topic_obj)

    # Stage 3 (text-only subset)
    duration_sec = float(payload.duration_sec) if payload.duration_sec else 60.0
    stage3 = {
        "fluency": {
            "duration_sec": round(duration_sec, 2),
            "wpm": calculate_wpm(transcript, duration_sec),
            "pause_ratio": 0.0,  # not available in text-only
            "filler_words": analyze_fillers(transcript),
        },
        "grammar": analyze_grammar(transcript),
    }

    # Stage 4
    stage4 = run_stage4(stage3)

    # Stage 5 (YOUR signature)
    stage5 = run_stage5(topic_obj=topic_obj, transcript=transcript)

    # Stage 6 (YOUR signature)
    stage6 = run_stage6(
        topic_text=topic_text,
        transcript=transcript,
        stage4_results=stage4,
        stage5_results=stage5,
        save_history=bool(payload.save_history),
    )

    return EvaluateTextResponse(
        topic_obj=topic_obj,
        topic_text=topic_text,
        transcript=transcript,
        stage3=stage3,
        stage4=stage4,
        stage5=stage5,
        stage6=stage6,
    )

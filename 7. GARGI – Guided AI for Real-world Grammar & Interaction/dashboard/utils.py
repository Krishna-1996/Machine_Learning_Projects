import json
import pandas as pd
from pathlib import Path

SESSIONS_FILE = Path("sessions/sessions.jsonl")

def load_sessions() -> pd.DataFrame:
    if not SESSIONS_FILE.exists():
        return pd.DataFrame()

    records = []
    with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        return pd.DataFrame()

    df = pd.json_normalize(records)

    # -------------------------------
    # Timestamp normalization
    # -------------------------------
    if "timestamp_utc" not in df.columns:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    # -------------------------------
    # Schema mapping (RAW → DASHBOARD)
    # -------------------------------
    df["scores.overall"] = df["overall_quality_score"]
    df["scores.fluency"] = df["fluency_score"]
    df["scores.grammar"] = df["grammar_score"]
    df["scores.fillers"] = df["fillers_score"]
    
    df["relevance.relevance_score"] = df["relevance_score"]
    df["confidence.confidence_score"] = df["confidence_score"]

    df["topic_raw"] = df["topic"]

    # -------------------------------
    # REAL evidence-driven error metrics
    # -------------------------------
    df["grammar.error_count"] = df.get("grammar_error_count", 0)
    df["fillers.total"] = df.get("filler_total", 0)

    # -------------------------------
    # Optional: placeholders to avoid future crashes
    # -------------------------------
    if "wpm" in df.columns:
        df["wpm"] = df["wpm"]
    if "pause_ratio" in df.columns:
        df["pause_ratio"] = df["pause_ratio"]

    
    return df

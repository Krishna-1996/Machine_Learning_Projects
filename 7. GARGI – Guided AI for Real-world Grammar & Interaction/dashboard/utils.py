import sys
from pathlib import Path
import json
import pandas as pd


# -------------------------------------------------
# Ensure project root is on sys.path (Streamlit-safe)
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.paths import sessions_file  # noqa: E402

SESSIONS_FILE = sessions_file()


def _robust_json_objects(text: str):
    """
    Yield JSON objects from a text blob that may contain:
    - proper JSONL (1 object per line)
    - concatenated JSON objects without newlines
    - extra whitespace between objects
    """
    dec = json.JSONDecoder()
    i = 0
    n = len(text)

    while i < n:
        # skip whitespace
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break

        try:
            obj, j = dec.raw_decode(text, i)
            yield obj
            i = j
        except json.JSONDecodeError:
            # If we can't decode at this position, advance one char to avoid infinite loop.
            i += 1


def load_sessions() -> pd.DataFrame:
    if not SESSIONS_FILE.exists():
        return pd.DataFrame()

    raw_text = SESSIONS_FILE.read_text(encoding="utf-8", errors="ignore")
    records = list(_robust_json_objects(raw_text))

    if not records:
        return pd.DataFrame()

    df = pd.json_normalize(records)

    if "timestamp_utc" not in df.columns:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    # Schema mapping (keeps your existing dashboard stable)
    df["topic_raw"] = df.get("topic")

    df["scores.overall"] = df.get("overall_quality_score")
    df["scores.fluency"] = df.get("fluency_score")
    df["scores.grammar"] = df.get("grammar_score")
    df["scores.fillers"] = df.get("fillers_score")

    df["relevance.relevance_score"] = df.get("relevance_score")
    df["relevance.label"] = df.get("relevance_label")
    df["relevance.on_topic_ratio"] = df.get("on_topic_sentence_ratio")

    df["confidence.confidence_score"] = df.get("confidence_score")
    df["confidence.label"] = df.get("confidence_label")

    # Evidence metrics
    df["evidence.wpm"] = df.get("wpm")
    df["evidence.pause_ratio"] = df.get("pause_ratio")
    df["evidence.error_density"] = df.get("error_density")
    df["grammar.error_count"] = df.get("grammar_error_count")
    df["fillers.total"] = df.get("filler_total")

    # Numeric coercion for charts
    for col in [
        "scores.overall", "scores.fluency", "scores.grammar", "scores.fillers",
        "relevance.relevance_score", "confidence.confidence_score",
        "evidence.wpm", "evidence.pause_ratio", "evidence.error_density",
        "grammar.error_count", "fillers.total"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("timestamp")

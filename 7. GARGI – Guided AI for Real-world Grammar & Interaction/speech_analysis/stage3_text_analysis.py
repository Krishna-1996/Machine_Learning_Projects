"""
Stage 3 (Text-only): Fluency, Grammar, Lexical Metrics
Used by /evaluate/text when audio is not available
Project: GARGI
Author: Krishna
"""

from typing import Dict
import re

# Optional grammar tool
try:
    import language_tool_python
    _LANGUAGE_TOOL_OK = True
except Exception:
    _LANGUAGE_TOOL_OK = False


FILLER_WORDS = [
    "um", "uh", "ah", "erm", "like", "you know",
    "so", "actually", "basically", "well", "okay", "ok"
]


def _tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower())


def _count_fillers(text: str) -> Dict[str, int]:
    t = text.lower()
    counts = {}
    for f in FILLER_WORDS:
        counts[f] = len(re.findall(r"\b" + re.escape(f) + r"\b", t))
    return counts


def run_stage3_text(transcript: str, duration_sec: float) -> Dict:
    """
    Build a COMPLETE Stage 3 object compatible with Stage 4 scoring.
    """

    transcript = (transcript or "").strip()

    if not transcript or duration_sec <= 0:
        return {
            "fluency": {
                "wpm": 0.0,
                "pause_ratio": 0.15,
                "filler_count": 0,
                "duration_sec": duration_sec,
            },
            "fillers": {},
            "grammar": {
                "error_count": 0,
                "error_rate": 0.0,
                "tool": "none",
            },
            "lexical": {
                "word_count": 0,
                "unique_word_ratio": 0.0,
            },
        }

    words = _tokenize(transcript)
    word_count = len(words)

    # --- WPM ---
    minutes = duration_sec / 60.0
    wpm = round(word_count / minutes, 2) if minutes > 0 else 0.0

    # --- Fillers ---
    filler_counts = _count_fillers(transcript)
    filler_total = sum(filler_counts.values())

    # --- Grammar ---
    grammar_errors = 0
    grammar_tool = "none"

    if _LANGUAGE_TOOL_OK:
        try:
            tool = language_tool_python.LanguageTool("en-US")
            matches = tool.check(transcript)
            grammar_errors = len(matches)
            grammar_tool = "languagetool"
        except Exception:
            grammar_errors = 0
            grammar_tool = "languagetool_failed"

    error_rate = round(grammar_errors / max(word_count, 1), 3)

    # --- Lexical richness ---
    unique_ratio = round(len(set(words)) / max(word_count, 1), 3)

    return {
        "fluency": {
            "wpm": wpm,
            "pause_ratio": 0.15,  # neutral default (no audio)
            "filler_count": filler_total,
            "duration_sec": duration_sec,
        },
        "fillers": filler_counts,
        "grammar": {
            "error_count": grammar_errors,
            "error_rate": error_rate,
            "tool": grammar_tool,
        },
        "lexical": {
            "word_count": word_count,
            "unique_word_ratio": unique_ratio,
        },
    }

# speech_analysis/stage3_text_analysis.py

import re
from typing import Dict

# Optional grammar tool
try:
    import language_tool_python
    _LANG_TOOL_AVAILABLE = True
except Exception:
    _LANG_TOOL_AVAILABLE = False


FILLER_WORDS = [
    "um", "uh", "erm", "ah", "like", "you know",
    "so", "actually", "basically", "okay", "ok"
]


def _count_fillers(text: str) -> Dict[str, int]:
    text_l = text.lower()
    counts = {}
    for filler in FILLER_WORDS:
        pattern = r"\b" + re.escape(filler) + r"\b"
        counts[filler] = len(re.findall(pattern, text_l))
    return counts


def _basic_tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower())


def run_stage3_text(transcript: str, duration_sec: float) -> Dict:
    """
    Build Stage 3 metrics using ONLY text + duration.
    This schema MUST be stable for Stage 4 scoring.
    """

    if not transcript or duration_sec <= 0:
        # Hard fail-safe (never return empty Stage3)
        return {
            "fluency": {
                "wpm": 0.0,
                "pause_ratio": 0.15,
                "filler_count": 0,
            },
            "fillers": {},
            "grammar": {
                "error_count": 0,
                "error_rate": 0.0,
                "tool": "none"
            },
            "lexical": {
                "word_count": 0,
                "unique_word_ratio": 0.0
            }
        }

    words = _basic_tokenize(transcript)
    word_count = len(words)

    # --- WPM ---
    minutes = duration_sec / 60.0
    wpm = round(word_count / minutes, 2) if minutes > 0 else 0.0

    # --- Fillers ---
    filler_counts = _count_fillers(transcript)
    total_fillers = sum(filler_counts.values())

    # --- Grammar ---
    grammar_errors = 0
    grammar_tool = "none"

    if _LANG_TOOL_AVAILABLE:
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
    unique_word_ratio = round(len(set(words)) / max(word_count, 1), 3)

    return {
        "fluency": {
            "wpm": wpm,
            # no audio → neutral assumed pause ratio
            "pause_ratio": 0.15,
            "filler_count": total_fillers
        },
        "fillers": filler_counts,
        "grammar": {
            "error_count": grammar_errors,
            "error_rate": error_rate,
            "tool": grammar_tool
        },
        "lexical": {
            "word_count": word_count,
            "unique_word_ratio": unique_word_ratio
        }
    }

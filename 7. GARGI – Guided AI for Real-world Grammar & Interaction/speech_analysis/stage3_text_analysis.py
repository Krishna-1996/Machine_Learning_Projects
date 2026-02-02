"""
Stage 3 (Text-only): Speech Analysis from transcript + duration
Project: GARGI
Author: Krishna

Why this exists:
- Android sends transcript + duration_sec (no audio file to API)
- Stage 4 scoring requires Stage 3 fields (wpm, pause_ratio, filler_words, grammar)
- This module avoids librosa/audio dependencies and runs fast on Cloud Run
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List

import requests

LANGUAGETOOL_URL = os.getenv("LANGUAGETOOL_URL", "http://localhost:8081/v2/check")

FILLER_WORDS: List[str] = [
    "um", "uh", "ah", "like", "you know", "i mean", "so", "well",
    "actually", "basically", "right", "just", "hmm", "er"
]


# -------------------------------
# Text metrics
# -------------------------------
def analyze_fillers(text: str) -> Dict[str, int]:
    text = (text or "").lower()
    counts: Dict[str, int] = {}
    for filler in FILLER_WORDS:
        pattern = r"\b" + re.escape(filler) + r"\b"
        matches = re.findall(pattern, text)
        if matches:
            counts[filler] = len(matches)
    return counts


def calculate_wpm(text: str, duration_sec: float) -> float:
    """
    WPM = words / minutes. If duration is too small, return 0 (safe).
    """
    text = (text or "").strip()
    words = len(text.split()) if text else 0

    if duration_sec is None or duration_sec < 2.0:
        return 0.0

    minutes = duration_sec / 60.0
    if minutes <= 0:
        return 0.0

    return round(words / minutes, 1)


def estimate_pause_ratio_text_only(text: str) -> float:
    """
    We do NOT have audio pauses in /evaluate/text.
    So we use a neutral default that won't dominate scoring.

    You can later improve this using:
    - punctuation density
    - long gaps from timestamps (if STT provides)
    """
    _ = text  # reserved for future improvements
    return 0.15


# -------------------------------
# Grammar (LanguageTool)
# -------------------------------
def analyze_grammar(text: str) -> dict:
    """
    Always returns a stable schema even if LanguageTool is unavailable.
    Keys align with your existing stage3_analysis.py schema.
    """
    text = (text or "").strip()
    total_words = len(text.split()) if text else 0

    fallback = {
        "total_errors": 0,
        "error_density": 0.0,
        "rules_count": {},
        "errors": [],
        "warning": None
    }

    if not text:
        return fallback

    try:
        resp = requests.post(
            LANGUAGETOOL_URL,
            data={"text": text, "language": "en-US"},
            timeout=6
        )
        resp.raise_for_status()
        data = resp.json()

        matches = data.get("matches", []) or []
        errors = []
        rules_count = {}

        for m in matches:
            rule_id = (m.get("rule") or {}).get("id", "UNKNOWN")
            msg = m.get("message", "")
            suggestions = [
                r.get("value")
                for r in (m.get("replacements") or [])
                if isinstance(r, dict) and "value" in r
            ]

            rules_count[rule_id] = rules_count.get(rule_id, 0) + 1
            errors.append({
                "rule": rule_id,
                "message": msg,
                "suggestions": suggestions[:5]
            })

        total_errors = len(errors)
        error_density = (total_errors / total_words) * 100 if total_words > 0 else 0.0

        return {
            "total_errors": total_errors,
            "error_density": round(error_density, 2),
            "rules_count": rules_count,
            "errors": errors,
            "warning": None
        }

    except Exception as e:
        fallback["warning"] = f"LanguageTool unavailable: {e}"
        return fallback


# -------------------------------
# Orchestrator (Text-only Stage 3)
# -------------------------------
def run_stage3_text(transcript: str, duration_sec: float | None) -> Dict[str, Any]:
    """
    Returns SAME SHAPE as audio stage3_analysis.run_stage3():
    {
      "fluency": {duration_sec, wpm, pause_ratio, filler_words},
      "grammar": {...}
    }
    """
    transcript = (transcript or "").strip()
    duration_sec = float(duration_sec or 0.0)

    fluency = {
        "duration_sec": round(duration_sec, 2),
        "wpm": calculate_wpm(transcript, duration_sec),
        "pause_ratio": round(estimate_pause_ratio_text_only(transcript), 2),
        "filler_words": analyze_fillers(transcript)
    }

    grammar = analyze_grammar(transcript)

    return {
        "fluency": fluency,
        "grammar": grammar
    }

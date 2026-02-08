"""
Stage 3 (Text-only): Speech Analysis from transcript + duration
Project: GARGI
Author: Krishna

Used by /evaluate/text where we do not have an audio file.
This produces the same *shape* as the audio stage3_analysis.run_stage3()
so Stage 4 scoring can work consistently.
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
    text = (text or "").strip()
    words = len(text.split()) if text else 0

    # Avoid nonsense for very short durations
    if duration_sec is None or float(duration_sec) < 2.0:
        return 0.0

    minutes = float(duration_sec) / 60.0
    if minutes <= 0:
        return 0.0

    return round(words / minutes, 1)


def estimate_pause_ratio_text_only(_: str) -> float:
    # No audio available; neutral default.
    return 0.15


def analyze_grammar(text: str) -> dict:
    """
    Stable schema even if LanguageTool is unavailable.
    Matches your existing stage3_analysis.py schema.
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


def run_stage3_text(transcript: str, duration_sec: float | None) -> Dict[str, Any]:
    transcript = (transcript or "").strip()
    duration_sec = float(duration_sec or 0.0)

    fluency = {
        "duration_sec": round(duration_sec, 2),
        "wpm": calculate_wpm(transcript, duration_sec),
        "pause_ratio": round(estimate_pause_ratio_text_only(transcript), 2),
        "filler_words": analyze_fillers(transcript),
    }

    grammar = analyze_grammar(transcript)

    return {
        "fluency": fluency,
        "grammar": grammar
    }

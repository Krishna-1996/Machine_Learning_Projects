# speech_analysis/stage3_text_analysis.py

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

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


def count_words(text: str) -> int:
    text = (text or "").strip()
    if not text:
        return 0
    return len(text.split())


def estimate_duration_if_missing(word_count: int, baseline_wpm: float = 130.0) -> float:
    """
    Fallback if Android forgets to send duration or sends too-small duration.
    """
    if word_count <= 0:
        return 0.0
    minutes = word_count / float(baseline_wpm)
    return round(minutes * 60.0, 2)


def calculate_wpm(word_count: int, duration_sec: float) -> float:
    if duration_sec is None or float(duration_sec) < 2.0:
        return 0.0
    minutes = float(duration_sec) / 60.0
    if minutes <= 0:
        return 0.0
    return round(word_count / minutes, 1)


def estimate_pause_ratio_text_only(_: str) -> float:
    # No audio here; keep a neutral default.
    return 0.15


def analyze_grammar(text: str) -> dict:
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


def run_stage3_text(transcript: str, duration_sec: Optional[float]) -> Dict[str, Any]:
    transcript = (transcript or "").strip()
    wc = count_words(transcript)

    dur = float(duration_sec or 0.0)
    if dur < 2.0:
        dur = estimate_duration_if_missing(wc)

    dur = round(dur, 2)

    fillers = analyze_fillers(transcript)
    pause_ratio = round(estimate_pause_ratio_text_only(transcript), 2)
    wpm = calculate_wpm(wc, dur)

    grammar = analyze_grammar(transcript)

    fluency = {
        "duration_sec": dur,
        "word_count": wc,
        "wpm": wpm,
        "pause_ratio": pause_ratio,
        "filler_words": fillers,
    }

    # IMPORTANT: Return BOTH flat + nested so every downstream stage works.
    return {
        # Flat canonical keys (Stage 4 reads these)
        "transcript": transcript,
        "duration_sec": dur,
        "word_count": wc,
        "wpm": wpm,
        "pause_ratio": pause_ratio,
        "filler_words": fillers,
        "grammar_errors": grammar.get("total_errors", 0),
        "grammar_raw": grammar,

        # Nested for UI/compat (your formatter reads these)
        "fluency": fluency,
        "grammar": grammar,
    }
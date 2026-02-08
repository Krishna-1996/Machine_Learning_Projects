"""
Stage 3: Speech Analysis (Fluency + Grammar)
Project: GARGI
Author: Krishna
"""

import librosa
import re
import requests
import os

LANGUAGETOOL_URL = os.getenv(
    "LANGUAGETOOL_URL",
    "http://localhost:8081/v2/check"
)

FILLER_WORDS = [
    "um", "uh", "ah", "like", "you know", "i mean", "so", "well",
    "actually", "basically", "right", "just", "hmm", "er"
]

# -------------------------------
# Audio
# -------------------------------
def load_audio(audio_path: str):
    y, sr = librosa.load(audio_path, sr=16000)
    duration = len(y) / sr
    return y, sr, duration

def analyze_pauses(y, sr, duration):
    intervals = librosa.effects.split(y, top_db=25)
    speech_time = sum((end - start) / sr for start, end in intervals)
    pause_time = duration - speech_time
    return pause_time / duration if duration > 0 else 0

# -------------------------------
# Text
# -------------------------------
def analyze_fillers(text: str):
    text = text.lower()
    counts = {}
    for filler in FILLER_WORDS:
        pattern = r"\b" + re.escape(filler) + r"\b"
        matches = re.findall(pattern, text)
        if matches:
            counts[filler] = len(matches)
    return counts

def calculate_wpm(text: str, duration: float):
    words = len(text.split())
    return round(words / (duration / 60), 1) if duration > 0 else 0.0

# -------------------------------
# Grammar
# -------------------------------
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
                if "value" in r
            ]

            rules_count[rule_id] = rules_count.get(rule_id, 0) + 1
            errors.append({
                "rule": rule_id,
                "message": msg,
                "suggestions": suggestions[:5]
            })

        total_errors = len(errors)
        error_density = (
            (total_errors / total_words) * 100
            if total_words > 0 else 0.0
        )

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
# Orchestrator
# -------------------------------
def analyze_speech(audio_path: str, transcript: str) -> dict:
    y, sr, duration = load_audio(audio_path)

    return {
        "fluency": {
            "duration_sec": round(duration, 2),
            "wpm": calculate_wpm(transcript, duration),
            "pause_ratio": round(analyze_pauses(y, sr, duration), 2),
            "filler_words": analyze_fillers(transcript)
        },
        "grammar": analyze_grammar(transcript)
    }

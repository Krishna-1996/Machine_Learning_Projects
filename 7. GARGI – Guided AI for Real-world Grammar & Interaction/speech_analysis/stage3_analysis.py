"""
Stage 3: Speech Analysis (Fluency + Grammar)
Project: GARGI
Author: Krishna
"""
import re
import requests
import os

LANGUAGETOOL_URL = os.getenv(
    "LANGUAGETOOL_URL",
    "http://localhost:8081/v2/check"
)

FILLER_WORDS = [
    "um", "uh", "ah", "like", "you know", "i mean", "so", "well"
]

def analyze_fillers(text: str):
    text = text.lower()
    counts = {}
    for filler in FILLER_WORDS:
        counts[filler] = len(re.findall(rf"\b{filler}\b", text))
    return {k: v for k, v in counts.items() if v > 0}

def calculate_wpm(text: str, duration: float | None):
    if not duration or duration <= 0:
        return 0.0
    words = len(text.split())
    return round(words / (duration / 60), 1)

def analyze_grammar(text: str):
    if not text.strip():
        return {"total_errors": 0, "error_density": 0.0}

    try:
        r = requests.post(
            LANGUAGETOOL_URL,
            data={"text": text, "language": "en-US"},
            timeout=5
        )
        data = r.json()
        errors = data.get("matches", [])
        density = (len(errors) / len(text.split())) * 100
        return {
            "total_errors": len(errors),
            "error_density": round(density, 2)
        }
    except Exception:
        return {"total_errors": 0, "error_density": 0.0}

def analyze_text_only(transcript: str, duration: float | None):
    return {
        "fluency": {
            "wpm": calculate_wpm(transcript, duration),
            "filler_words": analyze_fillers(transcript)
        },
        "grammar": analyze_grammar(transcript),
        "transcript": transcript
    }

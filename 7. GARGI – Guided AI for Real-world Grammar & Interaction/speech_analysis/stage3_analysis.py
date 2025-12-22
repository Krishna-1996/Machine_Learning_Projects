"""
Stage 3: Speech Analysis (Fluency + Grammar)
Project: GARGI
Author: Krishna
"""

import librosa
import re
import requests
import os

AUDIO_FILE = "speech.wav"
TRANSCRIPT_FILE = "transcription.txt"
LANGUAGETOOL_URL = "http://localhost:8081/v2/check"

FILLER_WORDS = [
    "um", "uh", "ah", "like", "you know", "i mean", "so", "well",
    "actually", "basically", "right", "just", "hmm", "er"
]

# -------------------------------
# Audio
# -------------------------------
def load_audio():
    y, sr = librosa.load(AUDIO_FILE, sr=16000)
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
def analyze_fillers(text):
    text = text.lower()
    counts = {}
    for filler in FILLER_WORDS:
        pattern = r"\b" + re.escape(filler) + r"\b"
        matches = re.findall(pattern, text)
        if matches:
            counts[filler] = len(matches)
    return counts

def calculate_wpm(text, duration):
    words = len(text.split())
    return round(words / (duration / 60), 1) if duration > 0 else 0

# -------------------------------
# Grammar
# -------------------------------
def analyze_grammar(text: str) -> dict:
    try:
        response = requests.post(
            LANGUAGETOOL_URL,
            data={"text": text, "language": "en-US"},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        # parse matches...
        return parsed_results
    except Exception as e:
        # Graceful fallback
        return {
            "errors": [],
            "error_count": 0,
            "error_density": 0.0,
            "rules_count": {},
            "warning": f"LanguageTool unavailable: {e}"
        }

# -------------------------------
# Orchestrator
# -------------------------------
def run_stage3():
    if not os.path.exists(TRANSCRIPT_FILE):
        raise FileNotFoundError("Transcript not found.")

    with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    y, sr, duration = load_audio()

    return {
        "fluency": {
            "duration_sec": round(duration, 2),
            "wpm": calculate_wpm(text, duration),
            "pause_ratio": round(analyze_pauses(y, sr, duration), 2),
            "filler_words": analyze_fillers(text)
        },
        "grammar": analyze_grammar(text)
    }

"""
Stage 3A: Speech Fluency Analysis
Author: Krishna
Project: GARGI
"""

import librosa
import numpy as np
import os
import logging
import re

AUDIO_FILE = "speech.wav"
TRANSCRIPT_FILE = "transcription.txt"

SILENCE_DB_THRESHOLD = 25  # dB
MIN_PAUSE_SEC = 0.3

FILLER_WORDS = [
    "um", "uh", "ah", "like", "you know", "i mean", "so", "well",
    "actually", "basically", "right", "just", "hmm", "er"
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------
# Load Audio
# -------------------------------
def load_audio(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Audio file not found.")
    y, sr = librosa.load(path, sr=16000)
    return y, sr

# -------------------------------
# Pause Analysis
# -------------------------------
def analyze_pauses(y, sr):
    intervals = librosa.effects.split(y, top_db=SILENCE_DB_THRESHOLD)

    total_duration = len(y) / sr
    speech_duration = sum((end - start) / sr for start, end in intervals)
    pause_duration = total_duration - speech_duration

    pause_ratio = pause_duration / total_duration if total_duration > 0 else 0

    return {
        "total_duration": total_duration,
        "pause_duration": pause_duration,
        "pause_ratio": pause_ratio
    }

# -------------------------------
# Filler Analysis
# -------------------------------
def analyze_fillers(text):
    text = text.lower()
    count = 0
    for filler in FILLER_WORDS:
        pattern = r"\b" + re.escape(filler) + r"\b"
        count += len(re.findall(pattern, text))
    return count

# -------------------------------
# WPM
# -------------------------------
def calculate_wpm(text, duration_sec):
    words = len(text.split())
    minutes = duration_sec / 60
    return round(words / minutes, 1) if minutes > 0 else 0

# -------------------------------
# Main
# -------------------------------
def main():
    try:
        y, sr = load_audio(AUDIO_FILE)

        with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
            text = f.read()

        pause_metrics = analyze_pauses(y, sr)
        filler_count = analyze_fillers(text)
        wpm = calculate_wpm(text, pause_metrics["total_duration"])

        logging.info("---- Stage 3A: Fluency Report ----")
        logging.info(f"Total Duration: {pause_metrics['total_duration']:.2f}s")
        logging.info(f"Pause Duration: {pause_metrics['pause_duration']:.2f}s")
        logging.info(f"Pause Ratio: {pause_metrics['pause_ratio']:.2f}")
        logging.info(f"Words Per Minute (WPM): {wpm}")
        logging.info(f"Filler Words Count: {filler_count}")

        logging.info("Stage 3A completed successfully.")

    except Exception as e:
        logging.error(f"Stage 3A failed: {e}")

if __name__ == "__main__":
    main()

"""
Stage 3: Speech Fluency Analysis
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

SILENCE_DB_THRESHOLD = 25
MIN_SILENCE_DURATION = 0.3

FILLER_WORDS = [
    "um", "uh", "ah", "like", "you know", "i mean", "so", "well",
    "actually", "basically", "right", "just", "hmm", "er"
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_audio(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError("Audio file not found.")
    y, sr = librosa.load(file_path, sr=16000)
    return y, sr

def detect_pauses(y, sr):
    pauses = librosa.effects.split(
        y,
        top_db=SILENCE_DB_THRESHOLD
    )

    total_duration = len(y) / sr
    speech_duration = sum((end - start) / sr for start, end in pauses)
    pause_duration = total_duration - speech_duration

    return pause_duration, total_duration

def analyze_fillers(text):
    text = text.lower()
    filler_count = 0
    for filler in FILLER_WORDS:
        pattern = r"\b" + re.escape(filler) + r"\b"
        filler_count += len(re.findall(pattern, text))
    return filler_count

def calculate_wpm(text, duration_sec):
    words = len(text.split())
    minutes = duration_sec / 60
    return words / minutes if minutes > 0 else 0

def main():
    try:
        y, sr = load_audio(AUDIO_FILE)

        with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
            text = f.read()

        pause_time, total_time = detect_pauses(y, sr)
        filler_count = analyze_fillers(text)
        wpm = calculate_wpm(text, total_time)

        logging.info("---- Fluency Report ----")
        logging.info(f"Total Duration: {total_time:.2f}s")
        logging.info(f"Total Pause Time: {pause_time:.2f}s")
        logging.info(f"Pause Ratio: {pause_time/total_time:.2f}")
        logging.info(f"Words Per Minute (WPM): {wpm:.1f}")
        logging.info(f"Filler Words Count: {filler_count}")

        logging.info("Stage 3 completed successfully.")

    except Exception as e:
        logging.error(f"Stage 3 failed: {e}")

if __name__ == "__main__":
    main()

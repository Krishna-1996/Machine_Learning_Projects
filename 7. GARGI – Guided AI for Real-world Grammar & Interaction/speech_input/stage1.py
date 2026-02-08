"""
Stage 1: Speech Input & Language Detection
Author: Krishna
Project: GARGI
"""

import sounddevice as sd
from scipy.io.wavfile import write
from langdetect import detect, LangDetectException
import numpy as np
import os
import logging

# -------------------------------
# Configuration
# -------------------------------
AUDIO_FILE = "speech.wav" 
SAMPLE_RATE = 16000 # Hz
DURATION = 1200  # seconds
WHISPER_MODEL_SIZE = "base"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------
# Load Whisper Model ONCE
# -------------------------------
logging.info("Loading Whisper model...")
WHISPER_MODEL = whisper.load_model(WHISPER_MODEL_SIZE)

# -------------------------------
# Record Audio
# -------------------------------
def record_audio(filename=AUDIO_FILE, duration=DURATION, fs=SAMPLE_RATE):
    logging.info(f"Recording audio for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    max_val = np.max(np.abs(audio))
    if max_val == 0:
        raise ValueError("No audio detected. Please speak louder.")

    audio_int16 = np.int16(audio / max_val * 32767)
    write(filename, fs, audio_int16)

    logging.info(f"Audio saved to {filename}")
    return filename

# -------------------------------
# Transcribe Audio
# -------------------------------
def transcribe_audio(audio_file):
    if not os.path.exists(audio_file):
        raise FileNotFoundError("Audio file not found.")

    logging.info("Transcribing audio...")
    result = WHISPER_MODEL.transcribe(audio_file)

    text = result.get("text", "").strip()
    if len(text) < 3:
        raise ValueError("Transcription too short or empty.")

    return text

# -------------------------------
# Detect Language
# -------------------------------
def detect_text_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

# -------------------------------
# Main Pipeline
# -------------------------------
def main():
    try:
        audio_file = record_audio()
        text = transcribe_audio(audio_file)

        language = detect_text_language(text)

        logging.info(f"Transcription: {text}")
        logging.info(f"Detected Language: {language}")

        # Save transcription for later stages
        with open("transcription.txt", "w", encoding="utf-8") as f:
            f.write(text)

        if language == "en":
            logging.info("Stage 1 completed successfully. Ready for Stage 2.")
        else:
            logging.warning("Please speak in English.")

    except Exception as e:
        logging.error(f"Stage 1 failed: {e}")

if __name__ == "__main__":
    main()

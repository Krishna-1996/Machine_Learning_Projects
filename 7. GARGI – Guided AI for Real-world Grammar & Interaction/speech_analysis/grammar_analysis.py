"""
Stage 3B: Grammar & Sentence Analysis
Author: Krishna
Project: GARGI
"""

import requests
import logging
import os
import re

TRANSCRIPT_FILE = "transcription.txt"
LANGUAGETOOL_URL = "http://localhost:8081/v2/check"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------
# Load Transcript
# -------------------------------
def load_transcript(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Transcript file not found.")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# -------------------------------
# Grammar Check
# -------------------------------
def check_grammar(text):
    response = requests.post(
        LANGUAGETOOL_URL,
        data={
            "text": text,
            "language": "en-US"
        }
    )
    response.raise_for_status()
    return response.json()["matches"]

# -------------------------------
# Sentence Analysis
# -------------------------------
def analyze_sentences(text):
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    total_sentences = len(sentences)
    total_words = len(text.split())

    avg_sentence_length = (
        total_words / total_sentences
        if total_sentences > 0 else 0
    )

    return total_sentences, avg_sentence_length

# -------------------------------
# Main
# -------------------------------
def main():
    try:
        text = load_transcript(TRANSCRIPT_FILE)

        grammar_errors = check_grammar(text)
        error_count = len(grammar_errors)

        total_sentences, avg_sentence_length = analyze_sentences(text)
        total_words = len(text.split())

        error_density = (
            (error_count / total_words) * 100
            if total_words > 0 else 0
        )

        logging.info("---- Stage 3B: Grammar Report ----")
        logging.info(f"Total Words: {total_words}")
        logging.info(f"Total Sentences: {total_sentences}")
        logging.info(f"Average Sentence Length: {avg_sentence_length:.2f}")
        logging.info(f"Grammar Errors: {error_count}")
        logging.info(f"Error Density: {error_density:.2f} errors / 100 words")

        logging.info("Stage 3B completed successfully.")

    except Exception as e:
        logging.error(f"Stage 3B failed: {e}")

if __name__ == "__main__":
    main()

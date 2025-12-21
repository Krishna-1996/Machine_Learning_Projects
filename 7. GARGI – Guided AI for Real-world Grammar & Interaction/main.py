"""
Main Orchestrator for GARGI
Author: Krishna
"""
import subprocess
import requests
import os
import sys


import logging
import time

from topic_generation.generate_topic import get_random_topic
from speech_input.stage1 import (
    record_audio,
    transcribe_audio,
    detect_text_language
)

from speech_analysis.stage3_analysis import run_stage3
from scoring_feedback.stage4_scoring import run_stage4
from topic_relevance.stage5_relevance import run_stage5


logging.basicConfig(level=logging.INFO)

TRANSCRIPT_FILE = "transcription.txt"


def main():
    logging.info("Welcome to GARGI")

    # -------------------------------------------------
    # Stage 2: Topic Selection
    # -------------------------------------------------
    category = input(
        "Enter topic category (or press Enter for random): "
    ).strip() or None

    topic_data = get_random_topic(category=category)
    topic_text = topic_data["topic"]

    print("\nYour Speaking Topic:")
    print(f"👉 {topic_text}\n")

    input("Press ENTER when you are ready to start speaking...")

    # -------------------------------------------------
    # Stage 1: Speech Input
    # -------------------------------------------------
    logging.info("Recording will start in 3 seconds...")
    time.sleep(3)

    audio_file = record_audio()
    transcript = transcribe_audio(audio_file)

    language = detect_text_language(transcript)
    if language != "en":
        logging.warning("Non-English detected. Session stopped.")
        return

    with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
        f.write(transcript)

    logging.info("Stage 1 completed successfully.")

    # -------------------------------------------------
    # Stage 3: Speech Analysis (Fluency + Grammar)
    # -------------------------------------------------
    stage3_results = run_stage3()

    # -------------------------------------------------
    # Stage 4: Scoring + Explainability (XAI)
    # -------------------------------------------------
    stage4_results = run_stage4(stage3_results)

    # -------------------------------------------------
    # Stage 5: Topic Relevance (Local MPNet Model)
    # -------------------------------------------------
    stage5_results = run_stage5(topic_text, transcript)

    # -------------------------------------------------
    # Final Output (User-Facing)
    # -------------------------------------------------
    print("\n================ GARGI Evaluation ================\n")

    print("🔹 Overall Quality Score:")
    print(f"   {stage4_results['scores']['overall']} / 10\n")

    print("🔹 Detailed Quality Scores:")
    print(f"   • Fluency: {stage4_results['scores']['fluency']}/10")
    print(f"   • Grammar: {stage4_results['scores']['grammar']}/10")
    print(f"   • Fillers: {stage4_results['scores']['fillers']}/10\n")

    print("🔹 Topic Relevance:")
    print(f"   • Relevance Score: {stage5_results['relevance_score']}")
    print(f"   • Label: {stage5_results['label']}")
    print(f"   • Semantic Similarity: {stage5_results['semantic_similarity']}")
    print(f"   • Coverage Score: {stage5_results['coverage_score']}")

    if stage5_results["key_matches"]:
        print(
            f"   • Key Matches: {', '.join(stage5_results['key_matches'])}"
        )
    else:
        print("   • Key Matches: None")
    print(f"   • Missing Keywords (top): {', '.join(stage5_results['missing_keywords']) if stage5_results['missing_keywords'] else 'None'}")

    print("\n🔹 Feedback:")
    for item in stage4_results["feedback"]:
        print(f"   - {item}")

    print("\n🔹 Explainability (Why these scores?):")
    print("Scoring Trace:")
    for k, v in stage4_results["scoring_trace"].items():
        print(f"  {k}: {v}")

    print("\nEvidence Used:")
    for k, v in stage4_results["evidence"].items():
        print(f"  {k}: {v}")

    print("\n🧠 Topic Relevance Explanation:")
    print(f"   {stage5_results['explanation']}")

    
    print("\n=================================================\n")

    logging.info("Session completed successfully.")


if __name__ == "__main__":
    main()

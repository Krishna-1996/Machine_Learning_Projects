"""
Main Orchestrator for GARGI
Author: Krishna
"""

import logging
import time

from topic_generation.generate_topic import get_random_topic
from speech_input.stage1 import record_audio, transcribe_audio, detect_text_language
from speech_analysis.stage3_analysis import run_stage3
from scoring_feedback.stage4_scoring import run_stage4
from topic_relevance.stage5_relevance import run_stage5
from coaching.stage6_coaching import run_stage6
from services.languagetool_service import ensure_languagetool

logging.basicConfig(level=logging.INFO)

TRANSCRIPT_FILE = "transcription.txt"


def safe_join(items):
    if not items:
        return "None"
    return ", ".join(items)


def print_section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60 + "\n")


def main():
    logging.info("Welcome to GARGI")

    # -------------------------------------------------
    # Stage 2: Topic Selection
    # -------------------------------------------------
    category = input("Enter topic category (or press Enter for random): ").strip() or None
    topic_obj = get_random_topic(category=category)

    topic_text = topic_obj.get("topic_raw", "").strip()
    if not topic_text:
        logging.error("Topic generation returned empty topic text.")
        return

    print_section("TOPIC")
    print(f"Your Speaking Topic:\n👉 {topic_text}")

    input("\nPress ENTER when you are ready to start speaking...")

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

    print_section("TRANSCRIPT (Saved to transcription.txt)")
    preview = transcript.strip().replace("\n", " ")
    if len(preview) > 350:
        preview = preview[:350] + "..."
    print(preview)

    LT_JAR = r"D:\Python Automation scripts\LanguageTool-6.6\LanguageTool-6.6\languagetool-server.jar"
    lt_ok = ensure_languagetool(jar_path=LT_JAR, port=8081)

    if not lt_ok:
        logging.warning("LanguageTool is not running. Grammar stage will run in fallback mode (no errors detected).")


    # -------------------------------------------------
    # Stage 3: Speech Analysis (Fluency + Grammar)
    # -------------------------------------------------
    stage3_results = run_stage3()

    # -------------------------------------------------
    # Stage 4: Scoring + Explainability (XAI)
    # -------------------------------------------------
    stage4_results = run_stage4(stage3_results)

    # -------------------------------------------------
    # Stage 5: Topic Relevance (Local MPNet)
    # -------------------------------------------------
    stage5_results = run_stage5(topic_obj, transcript)  # pass full topic object
    print_section("TOPIC RELEVANCE (Stage 5)")

    print("Relevance Metrics:")
    print(f"• Relevance Score: {stage5_results.get('relevance_score', 'N/A')}")
    print(f"• Label: {stage5_results.get('label', 'N/A')}")
    print(f"• Semantic Similarity: {stage5_results.get('semantic_similarity', 'N/A')}")
    cov_val = stage5_results.get("semantic_coverage", stage5_results.get("coverage_score", "N/A"))
    print(f"• Semantic Coverage: {cov_val}")
    print(f"• On-topic Sentence Ratio: {stage5_results.get('on_topic_sentence_ratio', 'N/A')}")

    print("\nExplainability:")
    print(f"• Topic Content Used: {stage5_results.get('topic_content', 'N/A')}")
    print(f"• Key Matches (topic concepts matched): {safe_join(stage5_results.get('key_matches', []))}")
    print(f"• Missing Concepts (top): {safe_join(stage5_results.get('missing_keywords', []))}")
    print(f"• Response Keyphrases (YAKE): {safe_join(stage5_results.get('response_keyphrases', []))}")

    print("\nRelevance Explanation:")
    print(stage5_results.get("explanation", "N/A"))

    # Print anchor rubric if available
    ar = stage5_results.get("anchor_rubric", {})
    if ar and ar.get("score") is not None:
        print("\nAnchor Rubric:")
        print(f"• Score: {ar.get('score')}")
        print(f"• Components: {ar.get('components')}")
        print(f"• Explanation: {ar.get('explanation')}")

    # -------------------------------------------------
    # Stage 6: Coaching + Trust + Progress Tracking
    # -------------------------------------------------
    stage6_results = run_stage6(
        topic_text,  # topic_raw
        transcript,
        stage4_results,
        stage5_results,
        save_history=True
    )
    print_section("LEARNING GUIDANCE & TRUST (Stage 6)")

    conf = stage6_results.get("confidence", {})
    print("Confidence:")
    print(f"• Confidence Score: {conf.get('confidence_score', 'N/A')}")
    print(f"• Confidence Label: {conf.get('confidence_label', 'N/A')}")
    print(f"• Explanation: {conf.get('confidence_explanation', 'N/A')}")

    print("\nTop Priorities (next attempt):")
    for i, p in enumerate(stage6_results.get("priorities", []), start=1):
        print(f"{i}. {p.get('area')} [{p.get('severity')}]")
        print(f"   Reason: {p.get('reason')}")
        print(f"   Action: {p.get('action')}")

    print("\nCoaching Feedback:")
    for line in stage6_results.get("coaching_feedback", []):
        print(f"- {line}")

    print("\nReflection Prompts:")
    for q in stage6_results.get("reflection_prompts", []):
        print(f"- {q}")

    log_path = stage6_results.get("history_log_path")
    if log_path:
        print(f"\nSession saved to: {log_path}")

    print("\n" + "=" * 60)
    logging.info("Session completed successfully.")


if __name__ == "__main__":
    main()

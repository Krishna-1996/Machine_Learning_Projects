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
    topic_data = get_random_topic(category=category)

    topic_text = topic_data.get("topic", "").strip()
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
    stage5_results = run_stage5(topic_text, transcript)
    print_section("TOPIC RELEVANCE (Stage 5)")

    print("Relevance Metrics:")
    print(f"• Relevance Score: {stage5_results.get('relevance_score', 'N/A')}")
    print(f"• Label: {stage5_results.get('label', 'N/A')}")
    print(f"• Semantic Similarity: {stage5_results.get('semantic_similarity', 'N/A')}")

    cov_val = stage5_results.get("semantic_coverage", stage5_results.get("coverage_score", "N/A"))
    print(f"• Semantic Coverage: {cov_val}")

    # New: % on-topic content
    print(f"• On-topic Sentence Ratio: {stage5_results.get('on_topic_sentence_ratio', 'N/A')}")

    print("\nExplainability:")
    print(f"• Topic Content Used: {stage5_results.get('topic_content', 'N/A')}")
    print(f"• Key Matches (topic concepts matched): {safe_join(stage5_results.get('key_matches', []))}")
    print(f"• Missing Concepts (top): {safe_join(stage5_results.get('missing_keywords', []))}")

    # New: YAKE response keyphrases
    print(f"• Response Keyphrases (YAKE): {safe_join(stage5_results.get('response_keyphrases', []))}")

    # New: Sentence-level breakdown (compact)
    print("\nSentence-level Similarities (compact):")
    sent_sims = stage5_results.get("sentence_similarities", [])
    if not sent_sims:
        print("• None")
    else:
        for i, item in enumerate(sent_sims[:6], start=1):  # show up to 6 sentences
            flag = "ON" if item.get("on_topic") else "OFF"
            simv = item.get("similarity", "N/A")
            s = item.get("sentence", "")
            if len(s) > 90:
                s = s[:90] + "..."
            print(f"• S{i} [{flag}] sim={simv}: {s}")

    print("\nRelevance Explanation:")
    print(stage5_results.get("explanation", "N/A"))


    # -------------------------------------------------
    # OUTPUT: Stage 4 (Quality)
    # -------------------------------------------------
    print_section("GARGI QUALITY EVALUATION (Stage 4)")

    scores = stage4_results.get("scores", {})
    print("Scores (0–10):")
    print(f"• Overall: {scores.get('overall', 'N/A')} / 10")
    print(f"• Fluency: {scores.get('fluency', 'N/A')} / 10")
    print(f"• Grammar: {scores.get('grammar', 'N/A')} / 10")
    print(f"• Fillers: {scores.get('fillers', 'N/A')} / 10")

    print("\nFeedback:")
    for item in stage4_results.get("feedback", []):
        print(f"- {item}")

    print("\nScoring Trace (XAI):")
    trace = stage4_results.get("scoring_trace", {})
    for dim, detail in trace.items():
        print(f"• {dim}: {detail}")

    print("\nEvidence Used:")
    evidence = stage4_results.get("evidence", {})
    for k, v in evidence.items():
        print(f"• {k}: {v}")

    # -------------------------------------------------
    # OUTPUT: Stage 5 (Relevance)
    # -------------------------------------------------
    print_section("TOPIC RELEVANCE (Stage 5)")

    print("Relevance Metrics:")
    print(f"• Relevance Score: {stage5_results.get('relevance_score', 'N/A')}")
    print(f"• Label: {stage5_results.get('label', 'N/A')}")
    print(f"• Semantic Similarity: {stage5_results.get('semantic_similarity', 'N/A')}")

    cov_val = stage5_results.get("semantic_coverage", stage5_results.get("coverage_score", "N/A"))
    print(f"• Semantic Coverage: {cov_val}")

    print("\nExplainability:")
    print(f"• Topic Content Used: {stage5_results.get('topic_content', 'N/A')}")
    print(f"• Key Matches (topic concepts matched): {safe_join(stage5_results.get('key_matches', []))}")
    print(f"• Missing Concepts (top): {safe_join(stage5_results.get('missing_keywords', []))}")
    print(f"• Response Keyphrases (what you talked about): {safe_join(stage5_results.get('response_keyphrases', []))}")

    print("\nRelevance Explanation:")
    print(stage5_results.get("explanation", "N/A"))

    print("\n" + "=" * 60)
    logging.info("Session completed successfully.")


if __name__ == "__main__":
    main()

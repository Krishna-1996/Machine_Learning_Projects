"""
Main Orchestrator for GARGI
Author: Krishna
"""

from topic_generation.generate_topic import get_random_topic
from speech_input.stage1 import record_audio, transcribe_audio, detect_text_language
from speech_analysis.fluency_analysis import analyze_pauses, analyze_fillers, calculate_wpm, load_audio
from grammer_analysis.grammar_analysis import load_transcript, check_grammar, analyze_sentences
import logging
import time

logging.basicConfig(level=logging.INFO)

def main():
    logging.info("Welcome to GARGI")

    # -------------------------------
    # Stage 2: Topic Selection
    # -------------------------------
    category = input("Enter topic category (or press Enter for random): ").strip() or None
    topic = get_random_topic(category=category)

    print("\nYour Speaking Topic:")
    print(f"👉 {topic['topic']}\n")

    input("Press ENTER when you are ready to start speaking...")

    # -------------------------------
    # Stage 1: Speech Input
    # -------------------------------
    logging.info("Recording will start in 3 seconds...")
    time.sleep(3)

    audio_file = record_audio()
    text = transcribe_audio(audio_file)

    if detect_text_language(text) != "en":
        logging.warning("Non-English detected. Stopping.")
        return
    try:
        # Save transcription for later stages
        with open("transcription.txt", "w", encoding="utf-8") as f:
            f.write(text)

        if language == "en":
            logging.info("Stage 1 completed successfully. Ready for Stage 2.")
        else:
            logging.warning("Please speak in English.")
    except Exception as e:
        logging.error(f"Stage 1 failed: {e}")

    # -------------------------------
    # Stage 3: Fluency Analysis
    # -------------------------------
    y, sr = load_audio(audio_file)
    pause_metrics = analyze_pauses(y, sr)
    filler_count = analyze_fillers(text)
    wpm = calculate_wpm(text, pause_metrics["total_duration"])

    print("\n--- Fluency Feedback ---")
    print(f"WPM: {wpm}")
    print(f"Pause Ratio: {pause_metrics['pause_ratio']:.2f}")
    print(f"Filler Words: {filler_count}")

    logging.info("Session completed successfully.")



    # -------------------------------
    # Stage 3B: Grammar & Sentence Analysis
    # -------------------------------
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

if __name__ == "__main__":
    main()

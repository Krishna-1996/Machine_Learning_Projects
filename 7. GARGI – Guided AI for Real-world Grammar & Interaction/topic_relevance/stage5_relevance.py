"""
Stage 5: Topic Relevance & Semantic Alignment
Project: GARGI
Author: Krishna
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# -------------------------------
# Load Local Model (Offline)
# -------------------------------

MODEL_PATH = r"D:\LLM Models\all-mpnet-base-v2"
model = SentenceTransformer(MODEL_PATH)


# -------------------------------
# Utility Functions
# -------------------------------

def clean_text(text):
    return re.sub(r"[^\w\s]", "", text.lower())


# -------------------------------
# Core Metrics
# -------------------------------

def semantic_similarity(topic, transcript):
    topic_emb = model.encode([topic], normalize_embeddings=True)
    transcript_emb = model.encode([transcript], normalize_embeddings=True)

    return float(cosine_similarity(topic_emb, transcript_emb)[0][0])


def keyword_coverage(topic, transcript):
    topic_words = set(clean_text(topic).split())
    transcript_words = set(clean_text(transcript).split())

    overlap = topic_words.intersection(transcript_words)
    coverage = len(overlap) / max(len(topic_words), 1)

    return round(coverage, 2), sorted(list(overlap))


def relevance_label(score):
    if score >= 0.85:
        return "Highly relevant"
    elif score >= 0.70:
        return "Mostly relevant"
    elif score >= 0.50:
        return "Partially relevant"
    else:
        return "Off-topic"


# -------------------------------
# Stage 5 Orchestrator
# -------------------------------

def run_stage5(topic, transcript):
    similarity = semantic_similarity(topic, transcript)
    coverage_score, matches = keyword_coverage(topic, transcript)

    final_score = round(0.6 * similarity + 0.4 * coverage_score, 2)

    explanation = (
        "Your response strongly aligns with the topic."
        if final_score >= 0.7
        else "Your response partially or weakly addresses the topic."
    )

    return {
        "relevance_score": final_score,
        "semantic_similarity": round(similarity, 2),
        "coverage_score": coverage_score,
        "key_matches": matches,
        "label": relevance_label(final_score),
        "explanation": explanation
    }

"""
Stage 5: Topic Relevance & Semantic Alignment (Improved Coverage + Explainability)
Project: GARGI
Author: Krishna
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# -------------------------------
# Local model path (offline)
# -------------------------------
MODEL_PATH = r"D:\LLM Models\all-mpnet-base-v2"
model = SentenceTransformer(MODEL_PATH)

# -------------------------------
# Simple stopwords list (extend anytime)
# -------------------------------
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else",
    "to", "of", "in", "on", "at", "for", "with", "from", "by", "as",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "it", "this", "that", "these", "those",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "her", "our", "their",
    "do", "does", "did", "doing",
    "have", "has", "had", "having",
    "can", "could", "will", "would", "should", "may", "might", "must",
    "not", "no", "yes",
    "so", "just", "very", "really", "also",
}

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)         # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()     # normalize spaces
    return text

def tokenize_meaningful(text: str):
    """Tokenize and remove stopwords + short tokens."""
    tokens = clean_text(text).split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= 3]
    return tokens

def semantic_similarity(topic: str, transcript: str) -> float:
    topic_emb = model.encode([topic], normalize_embeddings=True)
    transcript_emb = model.encode([transcript], normalize_embeddings=True)
    return float(cosine_similarity(topic_emb, transcript_emb)[0][0])

def keyword_coverage(topic: str, transcript: str):
    """
    Coverage based on meaningful token overlap.
    Returns:
      coverage_score (0..1), key_matches(list), missing_keywords(list)
    """
    topic_tokens = set(tokenize_meaningful(topic))
    transcript_tokens = set(tokenize_meaningful(transcript))

    if not topic_tokens:
        return 0.0, [], []

    overlap = sorted(list(topic_tokens.intersection(transcript_tokens)))
    missing = sorted(list(topic_tokens.difference(transcript_tokens)))

    coverage = len(overlap) / len(topic_tokens)
    return round(coverage, 2), overlap, missing

def relevance_label(score: float) -> str:
    if score >= 0.85:
        return "Highly relevant"
    elif score >= 0.70:
        return "Mostly relevant"
    elif score >= 0.50:
        return "Partially relevant"
    else:
        return "Off-topic"

def run_stage5(topic: str, transcript: str):
    sim = semantic_similarity(topic, transcript)
    coverage, key_matches, missing = keyword_coverage(topic, transcript)

    # Weighted relevance score (still simple & explainable)
    relevance = round(0.6 * sim + 0.4 * coverage, 2)

    # Better explanation for the user
    if relevance >= 0.85:
        explanation = "Your response strongly and directly addresses the topic."
    elif relevance >= 0.70:
        explanation = "Your response addresses the topic, but could include more specific topic details."
    elif relevance >= 0.50:
        explanation = "Your response is somewhat related, but parts may be off-topic or too general."
    else:
        explanation = "Your response appears largely off-topic compared to the prompt."

    return {
        "relevance_score": relevance,
        "semantic_similarity": round(sim, 2),
        "coverage_score": coverage,
        "key_matches": key_matches,
        "missing_keywords": missing[:10],  # keep output short (top 10)
        "label": relevance_label(relevance),
        "explanation": explanation
    }

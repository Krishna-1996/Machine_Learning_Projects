"""
Stage 5: Topic Relevance & Semantic Alignment (Robust + Explainable)
Project: GARGI
Author: Krishna

- Uses local SentenceTransformer model (offline)
- Removes instruction wrappers using patterns
- Uses semantic coverage (phrase-to-phrase embedding matching)
- Returns schema-stable keys:
    relevance_score, semantic_similarity,
    semantic_coverage (+ alias coverage_score),
    key_matches, missing_keywords, label, explanation
"""

from __future__ import annotations

import re
from typing import List, Tuple, Dict, Any

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Local model path (offline)
# -------------------------------
MODEL_PATH = r"D:\LLM Models\all-mpnet-base-v2"
model = SentenceTransformer(MODEL_PATH)

# -------------------------------
# Prompt wrapper patterns (extend when needed)
# -------------------------------
INSTRUCTION_PATTERNS = [
    r"^share\s+(tips|advice|ways|strategies)\s+(for|to)\s+",
    r"^give\s+(tips|advice|ways|strategies)\s+(for|to)\s+",
    r"^explain\s+",
    r"^describe\s+",
    r"^discuss\s+",
    r"^talk\s+about\s+",
    r"^tell\s+me\s+about\s+",
    r"^compare\s+",
    r"^contrast\s+",
    r"^argue\s+",
    r"^do\s+you\s+agree\s+or\s+disagree\s+that\s+",
    r"^do\s+you\s+agree\s+that\s+",
    r"^do\s+you\s+disagree\s+that\s+",
    r"^what\s+are\s+the\s+(advantages|disadvantages|pros|cons)\s+of\s+",
    r"^give\s+reasons\s+(for|why)\s+",
    r"^why\s+",
    r"^how\s+",
]

# -------------------------------
# Basic stopwords (extend anytime)
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

# -------------------------------
# Text utilities
# -------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)       # punctuation -> space
    text = re.sub(r"\s+", " ", text).strip()   # normalize spaces
    return text

def tokenize_meaningful(text: str) -> List[str]:
    tokens = clean_text(text).split()
    return [t for t in tokens if t not in STOPWORDS and len(t) >= 3]

def normalize_topic(topic: str) -> str:
    """
    Remove instruction wrapper while keeping topic content.
    If no pattern matches, return cleaned topic.
    """
    t = clean_text(topic)
    for pat in INSTRUCTION_PATTERNS:
        t_new = re.sub(pat, "", t).strip()
        if t_new != t and len(t_new) >= 3:
            return t_new
    return t

def ngrams(tokens: List[str], n: int) -> List[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def keyphrase_candidates(text: str) -> List[str]:
    """
    Candidate phrases: unigrams, bigrams, trigrams (lightweight).
    """
    tokens = tokenize_meaningful(text)
    cands = set(tokens)
    cands.update(ngrams(tokens, 2))
    cands.update(ngrams(tokens, 3))

    phrases = [c for c in cands if c and len(c.split()) <= 3]
    # prefer longer phrases for interpretability
    phrases.sort(key=lambda x: (-len(x.split()), x))
    return phrases

# -------------------------------
# Core metrics
# -------------------------------
def semantic_similarity(topic: str, transcript: str) -> float:
    topic_emb = model.encode([topic], normalize_embeddings=True)
    transcript_emb = model.encode([transcript], normalize_embeddings=True)
    return float(cosine_similarity(topic_emb, transcript_emb)[0][0])

def semantic_coverage(
    topic_content: str,
    transcript: str,
    match_threshold: float = 0.60,
    max_phrases: int = 40
) -> Tuple[float, List[str], List[str]]:
    """
    For each topic phrase, find best semantic match in transcript phrases.
    Coverage = fraction matched >= threshold.
    """
    topic_phrases = keyphrase_candidates(topic_content)[:max_phrases]
    resp_phrases = keyphrase_candidates(transcript)[:max_phrases]

    if not topic_phrases or not resp_phrases:
        return 0.0, [], topic_phrases[:15]

    topic_emb = model.encode(topic_phrases, normalize_embeddings=True)
    resp_emb = model.encode(resp_phrases, normalize_embeddings=True)

    sims = cosine_similarity(topic_emb, resp_emb)
    best = sims.max(axis=1)

    matched = [p for p, s in zip(topic_phrases, best) if s >= match_threshold]
    missing = [p for p, s in zip(topic_phrases, best) if s < match_threshold]

    coverage = len(matched) / max(len(topic_phrases), 1)
    return round(coverage, 2), matched[:15], missing[:15]

def relevance_label(score: float) -> str:
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
def run_stage5(topic: str, transcript: str) -> Dict[str, Any]:
    topic_content = normalize_topic(topic)

    sim = semantic_similarity(topic, transcript)
    cov, matched, missing = semantic_coverage(
        topic_content, transcript, match_threshold=0.60, max_phrases=40
    )

    # Robust weighting: similarity drives relevance; coverage adds interpretability
    relevance = round(0.8 * sim + 0.2 * cov, 2)
    label = relevance_label(relevance)

    # Explanation
    if label in ("Highly relevant", "Mostly relevant"):
        explanation = "Your response aligns well with the topic based on semantic similarity and concept coverage."
    elif label == "Partially relevant":
        explanation = "Your response is somewhat related, but key topic concepts appear missing or weakly addressed."
    else:
        explanation = "Your response appears largely off-topic relative to the prompt."

    return {
        "topic_content": topic_content,
        "relevance_score": relevance,
        "semantic_similarity": round(sim, 2),

        # Schema-stable coverage fields
        "semantic_coverage": cov,
        "coverage_score": cov,  # backward-compatible alias

        "key_matches": matched,
        "missing_keywords": missing,
        "label": label,
        "explanation": explanation,
        "config": {
            "model_path": MODEL_PATH,
            "match_threshold": 0.60,
            "weights": {"similarity": 0.8, "coverage": 0.2}
        }
    }

"""
Stage 5: Topic Relevance & Semantic Alignment (Robust + Explainable)
Project: GARGI
Author: Krishna

Upgrades:
1) YAKE keyphrase extraction for response explainability
2) Sentence-level on-topic ratio (measures % of content relevant)

Core scoring:
- relevance_score = 0.8 * semantic_similarity + 0.2 * semantic_coverage

Outputs (schema-stable):
- relevance_score, semantic_similarity
- semantic_coverage (+ alias coverage_score)
- key_matches, missing_keywords
- response_keyphrases (YAKE)
- on_topic_sentence_ratio, sentence_similarities
- label, explanation
"""

from __future__ import annotations

import re
from typing import List, Tuple, Dict, Any

import yake
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Local model path (offline)
# -------------------------------
MODEL_PATH = r"D:\LLM Models\all-mpnet-base-v2"
model = SentenceTransformer(MODEL_PATH)

# -------------------------------
# Prompt wrapper patterns (extend)
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
# Stopwords (simple)
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

# Words common in prompts that are not useful as "topic concepts"
FRAME_WORDS = {
    "describe", "explain", "discuss", "talk", "share", "tell", "compare", "contrast",
    "scientific", "science", "discovery", "discoveries", "excite", "excites", "exciting",
    "meaningful", "topic", "example"
}

# -------------------------------
# Text utilities
# -------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_topic(topic: str) -> str:
    t = clean_text(topic)
    for pat in INSTRUCTION_PATTERNS:
        t_new = re.sub(pat, "", t).strip()
        if t_new != t and len(t_new) >= 3:
            return t_new
    return t

def tokenize_meaningful(text: str) -> List[str]:
    tokens = clean_text(text).split()
    return [t for t in tokens if t not in STOPWORDS and len(t) >= 3]

def ngrams(tokens: List[str], n: int) -> List[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def dedupe_phrases(phrases: List[str]) -> List[str]:
    phrases = sorted(set(phrases), key=lambda x: (-len(x.split()), x))
    kept = []
    for p in phrases:
        if not any(p != k and p in k for k in kept):
            kept.append(p)
    return kept

def keyphrase_candidates(text: str, remove_frame_words: bool = True) -> List[str]:
    tokens = tokenize_meaningful(text)
    if remove_frame_words:
        tokens = [t for t in tokens if t not in FRAME_WORDS]

    cands = set(tokens)
    cands.update(ngrams(tokens, 2))
    cands.update(ngrams(tokens, 3))

    phrases = [c for c in cands if c and len(c.split()) <= 3]
    phrases = dedupe_phrases(phrases)
    phrases.sort(key=lambda x: (-len(x.split()), x))
    return phrases

# -------------------------------
# YAKE keyphrases for response explainability
# -------------------------------
def yake_keyphrases(text: str, top_k: int = 10) -> List[str]:
    """
    Extract meaningful keyphrases from the transcript using YAKE.
    """
    # YAKE expects natural text; do not heavily clean it
    kw_extractor = yake.KeywordExtractor(
        lan="en",
        n=3,
        dedupLim=0.9,
        top=top_k
    )
    keywords = kw_extractor.extract_keywords(text)
    # keywords = [(phrase, score), ...] -> lower score = more important
    phrases = [k[0].strip().lower() for k in keywords]
    # basic cleanup + de-dupe
    phrases = [re.sub(r"\s+", " ", p) for p in phrases if p]
    return list(dict.fromkeys(phrases))

# -------------------------------
# Sentence-level relevance (% on-topic)
# -------------------------------
def split_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter; robust enough for your current use.
    """
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    # remove very short noise fragments
    return [p.strip() for p in parts if len(p.strip().split()) >= 4]

def sentence_on_topic_ratio(topic: str, transcript: str, threshold: float = 0.60):
    """
    Returns:
      ratio (0..1), per_sentence_sims [{sentence, sim, on_topic}, ...]
    """
    sentences = split_sentences(transcript)
    if not sentences:
        return 0.0, []

    topic_emb = model.encode([topic], normalize_embeddings=True)
    sent_embs = model.encode(sentences, normalize_embeddings=True)

    sims = cosine_similarity(topic_emb, sent_embs)[0]  # shape: (num_sentences,)
    per = []
    on_topic_count = 0

    for s, sim in zip(sentences, sims):
        sim_f = float(sim)
        on_topic = sim_f >= threshold
        if on_topic:
            on_topic_count += 1
        per.append({
            "sentence": s,
            "similarity": round(sim_f, 2),
            "on_topic": on_topic
        })

    ratio = on_topic_count / len(sentences)
    return round(ratio, 2), per

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
    """
    topic_phrases = keyphrase_candidates(topic_content, remove_frame_words=True)[:max_phrases]
    resp_phrases = keyphrase_candidates(transcript, remove_frame_words=True)[:max_phrases]

    if not topic_phrases or not resp_phrases:
        return 0.0, [], topic_phrases[:10]

    topic_emb = model.encode(topic_phrases, normalize_embeddings=True)
    resp_emb = model.encode(resp_phrases, normalize_embeddings=True)

    sims = cosine_similarity(topic_emb, resp_emb)
    best = sims.max(axis=1)

    matched = [p for p, s in zip(topic_phrases, best) if s >= match_threshold]
    missing = [p for p, s in zip(topic_phrases, best) if s < match_threshold]

    coverage = len(matched) / max(len(topic_phrases), 1)
    return round(coverage, 2), matched[:10], missing[:10]

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
    cov, matched, missing = semantic_coverage(topic_content, transcript, match_threshold=0.60)

    # % on-topic (sentence-level)
    on_topic_ratio, per_sentence = sentence_on_topic_ratio(topic, transcript, threshold=0.60)

    # Response keyphrases (YAKE)
    response_phrases = yake_keyphrases(transcript, top_k=10)

    relevance = round(0.8 * sim + 0.2 * cov, 2)
    label = relevance_label(relevance)

    explanation = (
        "Relevance uses semantic similarity (topic vs full response) and semantic coverage of topic concepts. "
        "On-topic ratio is computed by sentence-level similarity to the prompt."
    )

    return {
        "topic_content": topic_content,

        "relevance_score": relevance,
        "semantic_similarity": round(sim, 2),

        # coverage fields (schema-stable)
        "semantic_coverage": cov,
        "coverage_score": cov,  # alias for compatibility

        # explainability
        "key_matches": matched,
        "missing_keywords": missing,
        "response_keyphrases": response_phrases,

        # your requested % relevance signal
        "on_topic_sentence_ratio": on_topic_ratio,
        "sentence_similarities": per_sentence,

        "label": label,
        "explanation": explanation,
        "config": {
            "model_path": MODEL_PATH,
            "match_threshold": 0.60,
            "weights": {"similarity": 0.8, "coverage": 0.2},
            "sentence_threshold": 0.60
        }
    }

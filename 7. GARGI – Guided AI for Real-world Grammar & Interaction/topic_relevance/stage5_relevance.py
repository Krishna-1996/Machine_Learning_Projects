"""
Stage 5: Topic Relevance & Semantic Alignment
Project: GARGI — Guided AI for Real-world General Interaction
Author: Krishna

Upgrades in this version:
- Topic focus normalization: strips instruction phrasing ("describe", "talk about", etc.)
- Sentence splitting fallback for punctuation-poor transcripts (ASR-style text)
- Dual similarity blend: topic_content + topic_focus (more robust for generic prompts)
- Dynamic sentence threshold based on topic specificity
- Slightly improved labeling thresholds for instruction-style prompts
"""

from __future__ import annotations

import os
import re
from typing import Dict, Any, List, Tuple

import yake
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_WINDOWS_MODEL_PATH = r"D:\LLM Models\all-mpnet-base-v2"
MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", DEFAULT_WINDOWS_MODEL_PATH)

model = SentenceTransformer(MODEL_PATH)


# -------------------------------
# Text normalization
# -------------------------------
_INSTRUCTION_PREFIXES = [
    r"^describe\s+",
    r"^talk\s+about\s+",
    r"^explain\s+",
    r"^tell\s+me\s+about\s+",
    r"^share\s+",
    r"^give\s+",
    r"^what\s+is\s+",
    r"^why\s+is\s+",
]

def normalize_topic_focus(topic: str) -> str:
    """
    Convert instruction-like topics into a focus phrase.
    Example:
      "Describe an amazing scientific fact" -> "an amazing scientific fact"
    """
    t = (topic or "").strip()
    if not t:
        return ""

    t_low = t.lower().strip()

    # Remove common instruction prefixes
    for pat in _INSTRUCTION_PREFIXES:
        t_low = re.sub(pat, "", t_low).strip()

    # Remove trailing punctuation
    t_low = re.sub(r"[.?!:;]+$", "", t_low).strip()

    # If it becomes too short, fall back to original lowercased topic
    if len(t_low.split()) < 2:
        return (topic or "").lower().strip()

    return t_low


# -------------------------------
# YAKE keyphrases
# -------------------------------
def yake_keyphrases(text: str, top_k: int = 10) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    kw = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=top_k)
    pairs = kw.extract_keywords(text)
    out, seen = [], set()
    for phrase, _score in pairs:
        p = phrase.strip().lower()
        p = re.sub(r"\s+", " ", p)
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


# -------------------------------
# Sentence utilities
# -------------------------------
def split_sentences(text: str) -> List[str]:
    """
    Robust splitting:
    1) If punctuation exists, split on sentence boundaries.
    2) Else split on newlines.
    3) Else chunk by length (~18-28 words).
    """
    text = (text or "").strip()
    if not text:
        return []

    # 1) punctuation-based
    if re.search(r"[.!?]", text):
        parts = re.split(r"(?<=[.!?])\s+", text)
        sents = [p.strip() for p in parts if p.strip()]
    else:
        # 2) newline-based
        parts = [p.strip() for p in text.splitlines() if p.strip()]
        if len(parts) >= 2:
            sents = parts
        else:
            # 3) chunk by words
            words = text.split()
            sents = []
            i = 0
            while i < len(words):
                chunk = words[i:i+24]
                sents.append(" ".join(chunk).strip())
                i += 24

    # Keep sentences that have some substance
    return [s for s in sents if len(s.split()) >= 6]


def semantic_similarity(a: str, b: str) -> float:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return 0.0
    ea = model.encode([a], normalize_embeddings=True)
    eb = model.encode([b], normalize_embeddings=True)
    return float(cosine_similarity(ea, eb)[0][0])


def sentence_on_topic_ratio(topic_content: str, transcript: str, threshold: float) -> Tuple[float, List[Dict[str, Any]]]:
    sents = split_sentences(transcript)
    if not sents:
        return 0.0, []

    te = model.encode([topic_content], normalize_embeddings=True)
    se = model.encode(sents, normalize_embeddings=True)
    sims = cosine_similarity(te, se)[0]

    per = []
    on = 0
    for s, v in zip(sents, sims):
        simv = float(v)
        ok = simv >= threshold
        on += int(ok)
        per.append({"sentence": s, "similarity": round(simv, 2), "on_topic": ok})

    return round(on / len(sents), 2), per


# -------------------------------
# Coverage: topic phrases vs response phrases
# -------------------------------
def semantic_coverage(topic_phrases: List[str], response_phrases: List[str], match_threshold: float = 0.60):
    topic_phrases = [p for p in (topic_phrases or []) if p]
    response_phrases = [p for p in (response_phrases or []) if p]
    if not topic_phrases or not response_phrases:
        return 0.0, [], topic_phrases[:10]

    te = model.encode(topic_phrases, normalize_embeddings=True)
    re_ = model.encode(response_phrases, normalize_embeddings=True)

    sims = cosine_similarity(te, re_)
    best = sims.max(axis=1)

    matched = [p for p, s in zip(topic_phrases, best) if float(s) >= match_threshold]
    missing = [p for p, s in zip(topic_phrases, best) if float(s) < match_threshold]

    cov = len(matched) / max(len(topic_phrases), 1)
    return round(cov, 2), matched[:10], missing[:10]


# -------------------------------
# Labeling
# -------------------------------
def relevance_label(score: float) -> str:
    # Slightly more forgiving for instruction-heavy prompts
    if score >= 0.82:
        return "Highly relevant"
    elif score >= 0.68:
        return "Mostly relevant"
    elif score >= 0.48:
        return "Partially relevant"
    else:
        return "Off-topic"


# -------------------------------
# Stage 5 Orchestrator
# -------------------------------
def run_stage5(topic_obj: Dict[str, Any], transcript: str) -> Dict[str, Any]:
    topic_raw = topic_obj.get("topic_raw", "") or ""
    topic_content = topic_obj.get("topic_content", topic_raw) or ""
    topic_type = topic_obj.get("topic_type", "general")
    expected_anchors = topic_obj.get("expected_anchors", [])
    topic_keyphrases = topic_obj.get("topic_keyphrases", [])

    transcript = (transcript or "").strip()

    # Topic focus normalization (critical for prompts like "Describe ...")
    topic_focus = normalize_topic_focus(topic_content)

    # Response keyphrases (YAKE)
    resp_phrases = yake_keyphrases(transcript, top_k=12)

    # Similarity: blend topic_content and topic_focus
    sim_full = semantic_similarity(topic_content, transcript)
    sim_focus = semantic_similarity(topic_focus, transcript)
    sim = round((0.55 * sim_full + 0.45 * sim_focus), 4)

    # Coverage: fallback to YAKE on topic_focus if topic_keyphrases missing
    effective_topic_phrases = topic_keyphrases
    if not effective_topic_phrases or len(effective_topic_phrases) < 3:
        effective_topic_phrases = yake_keyphrases(topic_focus or topic_content, top_k=10)

    cov, matched, missing = semantic_coverage(effective_topic_phrases, resp_phrases, match_threshold=0.60)

    # Dynamic sentence threshold: less strict if topic is generic/short
    topic_len = len((topic_focus or topic_content).split())
    if topic_type in ("general", "experience", "story"):
        base_th = 0.44
    elif topic_type in ("event", "opinion", "compare", "explain", "advice"):
        base_th = 0.52
    else:
        base_th = 0.46

    if topic_len <= 5:
        sent_th = max(0.40, base_th - 0.05)
    else:
        sent_th = base_th

    on_ratio, per_sentence = sentence_on_topic_ratio(topic_focus or topic_content, transcript, threshold=sent_th)

    # Anchor rubric (kept for schema compatibility; if you use anchors later)
    anchors = {"score": None, "components": {}, "explanation": "No anchor rubric for this topic type."}
    bonus = 0.0

    # Relevance score (weights tuned for instruction prompts)
    relevance = 0.62 * sim + 0.18 * cov + 0.20 * on_ratio + bonus

    # Guardrail: if similarity is moderate-high, do not under-score
    if sim >= 0.60:
        relevance = max(relevance, 0.70)  # pushes clear on-topic answers into "Mostly relevant"

    relevance = round(max(0.0, min(1.0, relevance)), 2)
    label = relevance_label(relevance)

    explanation = (
        "Relevance blends semantic similarity between topic instruction and transcript (topic_content + topic_focus), "
        "semantic coverage (topic phrases vs response phrases), and sentence-level on-topic ratio with a dynamic threshold."
    )

    return {
        "topic_raw": topic_raw,
        "topic_content": topic_content,
        "topic_type": topic_type,
        "expected_anchors": expected_anchors,
        "topic_keyphrases": effective_topic_phrases,

        "relevance_score": relevance,
        "semantic_similarity": round(float(sim), 2),
        "semantic_coverage": cov,
        "coverage_score": cov,

        "key_matches": matched,
        "missing_keywords": missing,

        "response_keyphrases": resp_phrases,

        "on_topic_sentence_ratio": on_ratio,
        "sentence_similarities": per_sentence,

        "anchor_rubric": anchors,

        # Debug fields (helps you validate why a score happened)
        "debug": {
            "topic_focus": topic_focus,
            "sim_full": round(sim_full, 2),
            "sim_focus": round(sim_focus, 2),
            "sentence_threshold_used": sent_th,
            "topic_len_words": topic_len,
        },

        "label": label,
        "explanation": explanation,
        "config": {
            "weights": {"similarity": 0.62, "coverage": 0.18, "sentence_ratio": 0.20, "anchor_bonus_max": 0.10},
            "match_threshold": 0.60,
            "sentence_threshold": sent_th,
            "topic_phrase_fallback": True,
            "similarity_guardrail": 0.60,
            "guardrail_min_relevance": 0.70,
        }
    }

"""
Stage 5: Topic Relevance & Semantic Alignment (Upgraded for "Event" prompts)
Project: GARGI — Guided AI for Real-world General Interaction
Author: Krishna

Key upgrades:
- Strong topic normalization (includes "share your perspective on ...")
- YAKE keyphrase extraction for BOTH topic and response (robust for short prompts)
- Semantic coverage computed between topic keyphrases and response keyphrases
- Sentence-level on-topic ratio (dynamic threshold by prompt specificity)
- Event-specificity rubric:
    - checks for time anchor, place anchor, and concrete event description markers
    - reduces false "Off-topic" for event prompts while still penalizing vague answers

Schema-stable outputs:
- relevance_score, semantic_similarity
- semantic_coverage (+ alias coverage_score)
- key_matches, missing_keywords
- response_keyphrases (YAKE), topic_keyphrases (YAKE)
- on_topic_sentence_ratio, sentence_similarities
- event_specificity (if event prompt)
- label, explanation, config
"""
from __future__ import annotations

import re
from typing import Dict, Any, List, Tuple

import yake
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_PATH = r"D:\LLM Models\all-mpnet-base-v2"
model = SentenceTransformer(MODEL_PATH)

# -------------------------------
# YAKE response phrases
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
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip().split()) >= 4]

def semantic_similarity(a: str, b: str) -> float:
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
# Coverage: topic_keyphrases vs response_keyphrases
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
# Rubric checks (anchors)
# -------------------------------
TIME_PATTERNS = [
    r"\b(last|past)\s+(week|month|year)\b",
    r"\b(this)\s+(week|month|year)\b",
    r"\brecently\b",
    r"\byesterday\b",
    r"\btoday\b",
    r"\bin\s+20\d{2}\b",
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
]

EVENT_MARKERS = {"announced", "passed", "voted", "elected", "resigned", "protested", "signed", "sanction", "approved", "rejected"}
PLACE_HINTS = {"india", "bangladesh", "pakistan", "uk", "united kingdom", "usa", "united states", "china", "russia", "europe"}

def anchor_score(expected_anchors: List[str], transcript: str) -> Dict[str, Any]:
    """
    Heuristic presence checks for anchors. Not a fact checker.
    """
    t = (transcript or "").lower()
    components = {}

    if "time" in expected_anchors:
        components["time"] = any(re.search(p, t) for p in TIME_PATTERNS)
    if "place" in expected_anchors:
        components["place"] = any(ph in t for ph in PLACE_HINTS)
    if "what_happened" in expected_anchors:
        components["what_happened"] = any(m in t for m in EVENT_MARKERS)
    if "your_view" in expected_anchors or "position" in expected_anchors:
        components["your_view"] = any(p in t for p in ["i think", "i believe", "in my view", "my perspective", "i feel", "i support", "i disagree"])
    if "example" in expected_anchors:
        components["example"] = any(p in t for p in ["for example", "for instance", "such as"])

    # score
    if not components:
        return {"score": None, "components": {}, "explanation": "No anchor rubric for this topic type."}

    score = sum(1 for v in components.values() if v) / max(len(components), 1)
    score = round(score, 2)

    missing = [k for k, v in components.items() if not v]
    explanation = "Anchor coverage missing: " + ", ".join(missing) if missing else "Anchors well covered."

    return {"score": score, "components": components, "explanation": explanation}

# -------------------------------
# Labeling
# -------------------------------
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
# Stage 5 Orchestrator (metadata-aware)
# -------------------------------
def run_stage5(topic_obj: Dict[str, Any], transcript: str) -> Dict[str, Any]:
    topic_raw = topic_obj.get("topic_raw", "")
    topic_content = topic_obj.get("topic_content", topic_raw)
    topic_type = topic_obj.get("topic_type", "general")
    expected_anchors = topic_obj.get("expected_anchors", [])
    topic_keyphrases = topic_obj.get("topic_keyphrases", [])

    # Response keyphrases (YAKE)
    resp_phrases = yake_keyphrases(transcript, top_k=10)

    # Similarity should use topic_content (meaning), not topic_raw (instruction)
    sim = semantic_similarity(topic_content, transcript)

    # Coverage uses precomputed topic keyphrases (stable)
    cov, matched, missing = semantic_coverage(topic_keyphrases, resp_phrases, match_threshold=0.60)

    # Sentence ratio (dynamic threshold by type)
    # generic experience/advice prompts are broader => lower threshold
    if topic_type in ("general", "experience", "story"):
        sent_th = 0.50
    elif topic_type in ("event", "opinion", "compare", "explain", "advice"):
        sent_th = 0.55
    else:
        sent_th = 0.50

    on_ratio, per_sentence = sentence_on_topic_ratio(topic_content, transcript, threshold=sent_th)

    # Anchor rubric bonus (small)
    anchors = anchor_score(expected_anchors, transcript)
    bonus = 0.0
    if anchors["score"] is not None:
        bonus = 0.10 * float(anchors["score"])  # max +0.10

    # Final relevance score (balanced)
    relevance = 0.65 * sim + 0.15 * cov + 0.20 * on_ratio + bonus
    relevance = round(max(0.0, min(1.0, relevance)), 2)

    label = relevance_label(relevance)

    explanation = (
        "Relevance uses semantic similarity against topic_content, semantic coverage using topic_keyphrases, "
        "sentence-level on-topic ratio, and a small anchor bonus based on expected_anchors."
    )

    return {
        "topic_raw": topic_raw,
        "topic_content": topic_content,
        "topic_type": topic_type,
        "expected_anchors": expected_anchors,
        "topic_keyphrases": topic_keyphrases,

        "relevance_score": relevance,
        "semantic_similarity": round(sim, 2),
        "semantic_coverage": cov,
        "coverage_score": cov,  # alias

        "key_matches": matched,
        "missing_keywords": missing,

        "response_keyphrases": resp_phrases,

        "on_topic_sentence_ratio": on_ratio,
        "sentence_similarities": per_sentence,

        "anchor_rubric": anchors,

        "label": label,
        "explanation": explanation,
        "config": {
            "weights": {"similarity": 0.65, "coverage": 0.15, "sentence_ratio": 0.20, "anchor_bonus_max": 0.10},
            "match_threshold": 0.60,
            "sentence_threshold": sent_th
        }
    }

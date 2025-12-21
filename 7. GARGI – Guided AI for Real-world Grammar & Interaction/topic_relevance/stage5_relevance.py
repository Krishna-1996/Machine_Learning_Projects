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
from typing import List, Dict, Any, Tuple

import yake
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# Local model path (offline)
# -------------------------------
MODEL_PATH = r"D:\LLM Models\all-mpnet-base-v2"
model = SentenceTransformer(MODEL_PATH)


# -------------------------------
# Prompt wrapper patterns
# -------------------------------
INSTRUCTION_PATTERNS = [
    # High-value additions for your current prompt style:
    r"^share\s+your\s+perspective\s+on\s+",
    r"^share\s+your\s+view\s+on\s+",
    r"^share\s+your\s+opinion\s+on\s+",

    # Existing patterns
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


# Words common in prompts; we do NOT remove these from similarity,
# but we may avoid showing them as "concepts" in explainability.
FRAME_WORDS_FOR_DISPLAY = {
    "share", "discuss", "describe", "explain", "talk", "tell",
    "perspective", "opinion", "view",
    "recent", "event"
}


# -------------------------------
# Text utilities
# -------------------------------
def clean_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_topic(topic: str) -> str:
    """
    Remove instruction wrappers while keeping actual topic content.
    """
    t = clean_text(topic)
    for pat in INSTRUCTION_PATTERNS:
        t_new = re.sub(pat, "", t).strip()
        if t_new != t and len(t_new) >= 3:
            return t_new
    return t


def tokenize_meaningful(text: str) -> List[str]:
    tokens = clean_text(text).split()
    return [t for t in tokens if t not in STOPWORDS and len(t) >= 3]


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip().split()) >= 4]


# -------------------------------
# YAKE keyphrases (topic + response)
# -------------------------------
def yake_keyphrases(text: str, top_k: int = 10) -> List[str]:
    """
    Extract keyphrases using YAKE.
    Returns phrases lowercased; lower score => more important.
    """
    text = (text or "").strip()
    if not text:
        return []

    kw_extractor = yake.KeywordExtractor(
        lan="en",
        n=3,
        dedupLim=0.9,
        top=top_k
    )
    keywords = kw_extractor.extract_keywords(text)
    phrases = [k[0].strip().lower() for k in keywords if k and k[0]]
    phrases = [re.sub(r"\s+", " ", p) for p in phrases if p]
    # de-dupe while keeping order
    seen = set()
    out = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def filter_display_phrases(phrases: List[str]) -> List[str]:
    """
    For readability: remove prompt frame words from phrases like 'recent event'
    when showing missing/matched lists.
    """
    out = []
    for p in phrases:
        toks = p.split()
        toks = [t for t in toks if t not in FRAME_WORDS_FOR_DISPLAY]
        p2 = " ".join(toks).strip()
        if p2:
            out.append(p2)
    # de-dupe
    seen = set()
    final = []
    for p in out:
        if p not in seen:
            seen.add(p)
            final.append(p)
    return final


# -------------------------------
# Core semantic metrics
# -------------------------------
def semantic_similarity(topic: str, transcript: str) -> float:
    topic_emb = model.encode([topic], normalize_embeddings=True)
    tr_emb = model.encode([transcript], normalize_embeddings=True)
    return float(cosine_similarity(topic_emb, tr_emb)[0][0])


def semantic_coverage_keyphrases(
    topic_phrases: List[str],
    response_phrases: List[str],
    match_threshold: float = 0.60
) -> Tuple[float, List[str], List[str]]:
    """
    Coverage between YAKE topic phrases and YAKE response phrases using embeddings.
    Returns:
      coverage_score, matched_topic_phrases, missing_topic_phrases
    """
    topic_phrases = [p for p in topic_phrases if p]
    response_phrases = [p for p in response_phrases if p]

    if not topic_phrases or not response_phrases:
        return 0.0, [], topic_phrases[:10]

    t_emb = model.encode(topic_phrases, normalize_embeddings=True)
    r_emb = model.encode(response_phrases, normalize_embeddings=True)

    sims = cosine_similarity(t_emb, r_emb)
    best = sims.max(axis=1)

    matched = [p for p, s in zip(topic_phrases, best) if float(s) >= match_threshold]
    missing = [p for p, s in zip(topic_phrases, best) if float(s) < match_threshold]

    coverage = len(matched) / max(len(topic_phrases), 1)
    return round(coverage, 2), matched[:10], missing[:10]


def sentence_on_topic_ratio(topic_content: str, transcript: str, threshold: float) -> Tuple[float, List[Dict[str, Any]]]:
    sentences = split_sentences(transcript)
    if not sentences:
        return 0.0, []

    topic_emb = model.encode([topic_content], normalize_embeddings=True)
    sent_embs = model.encode(sentences, normalize_embeddings=True)
    sims = cosine_similarity(topic_emb, sent_embs)[0]

    per = []
    on_count = 0
    for s, sim in zip(sentences, sims):
        simv = float(sim)
        on = simv >= threshold
        if on:
            on_count += 1
        per.append({
            "sentence": s,
            "similarity": round(simv, 2),
            "on_topic": on
        })

    ratio = on_count / len(sentences)
    return round(ratio, 2), per


# -------------------------------
# Event prompt rubric (specificity)
# -------------------------------
EVENT_WORDS = {"event", "incident", "election", "vote", "law", "bill", "protest", "conflict", "summit", "speech", "court", "decision", "policy"}

TIME_PATTERNS = [
    r"\b(last|past)\s+(week|month|year)\b",
    r"\b(this)\s+(week|month|year)\b",
    r"\brecently\b",
    r"\byesterday\b",
    r"\btoday\b",
    r"\bin\s+20\d{2}\b",
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
]

PLACE_HINTS = {
    "india", "bangladesh", "pakistan", "uk", "united kingdom", "usa", "united states",
    "europe", "china", "russia", "israel", "gaza", "palestine", "ukraine"
}

EVENT_DESCRIPTION_MARKERS = {
    "announced", "passed", "voted", "elected", "resigned", "protested", "signed", "ban", "sanction",
    "released", "arrested", "declared", "approved", "rejected", "launched"
}


def is_event_prompt(topic_content: str) -> bool:
    t = clean_text(topic_content)
    toks = set(t.split())
    return len(toks.intersection(EVENT_WORDS)) > 0


def event_specificity_score(topic_content: str, transcript: str) -> Dict[str, Any]:
    """
    Measures whether the response anchors an "event" with enough specifics.
    Not a fact checker—only checks presence of anchors (time/place/what happened).
    """
    tr = clean_text(transcript)

    # time anchor
    time_hit = any(re.search(pat, tr) for pat in TIME_PATTERNS)

    # place anchor (simple heuristic; no NER dependency)
    place_hit = any(ph in tr for ph in PLACE_HINTS)

    # event description markers
    desc_hit = any(m in tr for m in EVENT_DESCRIPTION_MARKERS)

    # event noun presence
    event_noun_hit = any(w in tr.split() for w in EVENT_WORDS)

    # score: 0..1
    components = {
        "time_anchor": bool(time_hit),
        "place_anchor": bool(place_hit),
        "event_marker": bool(desc_hit),
        "event_noun": bool(event_noun_hit),
    }

    raw = sum(1 for v in components.values() if v) / 4.0
    score = round(raw, 2)

    # feedback
    missing = [k for k, v in components.items() if not v]
    explanation = "Event specificity checks for time/place/what-happened anchors. Missing: " + ", ".join(missing) if missing else "Good event anchoring detected."

    return {
        "score": score,
        "components": components,
        "explanation": explanation
    }


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
# Stage 5 Orchestrator
# -------------------------------
def run_stage5(topic: str, transcript: str) -> Dict[str, Any]:
    # Normalize topic to content
    topic_content = normalize_topic(topic)

    # YAKE phrases
    topic_phrases_raw = yake_keyphrases(topic_content, top_k=8)  # topic usually short; keep smaller
    resp_phrases_raw = yake_keyphrases(transcript, top_k=10)

    # Semantic similarity (prompt vs response)
    sim = semantic_similarity(topic, transcript)

    # Semantic coverage (topic keyphrases vs response keyphrases)
    cov, matched, missing = semantic_coverage_keyphrases(
        topic_phrases_raw, resp_phrases_raw, match_threshold=0.60
    )

    # Dynamic sentence threshold (generic short topics need lower threshold)
    t_tokens = tokenize_meaningful(topic_content)
    sent_threshold = 0.50 if len(t_tokens) <= 6 else 0.60
    on_ratio, per_sentence = sentence_on_topic_ratio(topic_content, transcript, threshold=sent_threshold)

    # Event specificity rubric (only if event prompt)
    event_spec = None
    event_bonus = 0.0
    if is_event_prompt(topic_content):
        event_spec = event_specificity_score(topic_content, transcript)
        # bonus is modest; it prevents unfair "off-topic" for valid but vague event framing
        event_bonus = 0.10 * float(event_spec["score"])  # max +0.10

    # Final relevance formula
    # similarity is core; coverage + sentence ratio confirm topical focus; event bonus helps event prompts.
    relevance = (
        0.65 * float(sim) +
        0.15 * float(cov) +
        0.20 * float(on_ratio) +
        float(event_bonus)
    )
    relevance = round(max(0.0, min(1.0, relevance)), 2)

    label = relevance_label(relevance)

    # Improve display of matches/missing by removing frame words
    matched_disp = filter_display_phrases(matched)
    missing_disp = filter_display_phrases(missing)

    # Explanation
    expl = (
        "Relevance uses semantic similarity (prompt vs response), semantic coverage (topic keyphrases vs response keyphrases), "
        "and sentence-level on-topic ratio. For event-style prompts, an additional specificity check rewards time/place/what-happened anchors."
    )

    return {
        "topic_content": topic_content,

        "relevance_score": relevance,
        "semantic_similarity": round(float(sim), 2),

        # coverage fields (schema-stable)
        "semantic_coverage": cov,
        "coverage_score": cov,  # alias

        # explainability fields
        "topic_keyphrases": topic_phrases_raw,
        "response_keyphrases": resp_phrases_raw,

        "key_matches": matched_disp,
        "missing_keywords": missing_disp,

        # sentence level
        "on_topic_sentence_ratio": on_ratio,
        "sentence_similarities": per_sentence,

        # event rubric
        "event_specificity": event_spec,  # None if not event prompt

        "label": label,
        "explanation": expl,
        "config": {
            "model_path": MODEL_PATH,
            "weights": {"similarity": 0.65, "coverage": 0.15, "sentence_ratio": 0.20, "event_bonus_max": 0.10},
            "match_threshold": 0.60,
            "sentence_threshold": sent_threshold,
            "topic_phrases_top_k": 8,
            "response_phrases_top_k": 10
        }
    }

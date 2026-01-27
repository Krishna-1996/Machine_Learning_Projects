"""
Stage 5: Topic Relevance & Semantic Alignment (Cloud-Run Ready)
Project: GARGI — Guided AI for Real-world General Interaction
Author: Krishna

Cloud upgrades:
- Removes hard dependency on sentence-transformers/torch/sklearn for Cloud Run.
- Uses Vertex AI embeddings when EMBEDDINGS_PROVIDER=vertex (default recommended).
- Keeps optional local provider for development (EMBEDDINGS_PROVIDER=local).
- Uses NumPy cosine similarity (no sklearn).

Recommended env vars (Cloud Run):
- EMBEDDINGS_PROVIDER=vertex
- VERTEX_LOCATION=asia-south1
- VERTEX_EMBED_MODEL=text-embedding-004

Project ID:
- Prefer GOOGLE_CLOUD_PROJECT if provided
- Otherwise auto-detect via google.auth.default() (ADC)
"""

from __future__ import annotations

import os
import re
from typing import Dict, Any, List, Tuple

import numpy as np
import yake


EMBEDDINGS_PROVIDER = (os.getenv("EMBEDDINGS_PROVIDER", "vertex") or "vertex").lower().strip()


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def _cosine_sim_vec(a: np.ndarray, b: np.ndarray) -> float:
    a = _l2_normalize(a.astype(np.float32))
    b = _l2_normalize(b.astype(np.float32))
    return float(np.dot(a, b))


def _cosine_sim_matrix(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    """
    one: (d,)
    many: (n, d)
    returns: (n,)
    """
    one = one.astype(np.float32)
    many = many.astype(np.float32)

    one = one / (np.linalg.norm(one) + 1e-12)
    many = many / (np.linalg.norm(many, axis=1, keepdims=True) + 1e-12)
    return np.dot(many, one)  # (n,)


def _get_project_id() -> str:
    # 1) explicit env var (best for clarity)
    pid = (os.getenv("GOOGLE_CLOUD_PROJECT", "") or "").strip()
    if pid:
        return pid

    # 2) ADC auto-detection (works in Cloud Run with service account)
    try:
        import google.auth  # lazy import
        _creds, pid2 = google.auth.default()
        pid2 = (pid2 or "").strip()
        if pid2:
            return pid2
    except Exception:
        pass

    raise RuntimeError(
        "Could not determine GCP project id. Set GOOGLE_CLOUD_PROJECT env var "
        "or ensure ADC works for the Cloud Run service account."
    )


def _embed_vertex(texts: List[str]) -> np.ndarray:
    """
    Returns embeddings as np.ndarray shape (n, d)
    Uses Cloud Run service account via ADC.
    """
    import vertexai  # lazy import
    from vertexai.language_models import TextEmbeddingModel  # lazy import

    project_id = _get_project_id()
    location = (os.getenv("VERTEX_LOCATION", "asia-south1") or "asia-south1").strip()
    model_name = (os.getenv("VERTEX_EMBED_MODEL", "text-embedding-004") or "text-embedding-004").strip()

    vertexai.init(project=project_id, location=location)
    model = TextEmbeddingModel.from_pretrained(model_name)

    embs = model.get_embeddings(texts)
    arr = np.array([e.values for e in embs], dtype=np.float32)
    return arr


def _embed_local(texts: List[str]) -> np.ndarray:
    """
    Local embeddings using SentenceTransformer.
    Do NOT use this on Cloud Run.
    """
    from sentence_transformers import SentenceTransformer  # lazy import

    model_path = (os.getenv("EMBEDDING_MODEL_PATH", "") or "").strip()
    if not model_path:
        raise RuntimeError("EMBEDDING_MODEL_PATH is required when EMBEDDINGS_PROVIDER=local.")

    model = SentenceTransformer(model_path)
    vecs = model.encode(texts, normalize_embeddings=True)
    return np.array(vecs, dtype=np.float32)


def embed_texts(texts: List[str]) -> np.ndarray:
    texts = [(t or "").strip() for t in (texts or [])]
    texts = [t for t in texts if t]
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    if EMBEDDINGS_PROVIDER == "local":
        return _embed_local(texts)

    return _embed_vertex(texts)


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
    t = (topic or "").strip()
    if not t:
        return ""

    t_low = t.lower().strip()
    for pat in _INSTRUCTION_PREFIXES:
        t_low = re.sub(pat, "", t_low).strip()

    t_low = re.sub(r"[.?!:;]+$", "", t_low).strip()

    if len(t_low.split()) < 2:
        return (topic or "").lower().strip()

    return t_low


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


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    if re.search(r"[.!?]", text):
        parts = re.split(r"(?<=[.!?])\s+", text)
        sents = [p.strip() for p in parts if p.strip()]
    else:
        parts = [p.strip() for p in text.splitlines() if p.strip()]
        if len(parts) >= 2:
            sents = parts
        else:
            words = text.split()
            sents = []
            i = 0
            while i < len(words):
                chunk = words[i : i + 24]
                sents.append(" ".join(chunk).strip())
                i += 24

    return [s for s in sents if len(s.split()) >= 6]


def semantic_similarity(a: str, b: str) -> float:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return 0.0

    embs = embed_texts([a, b])
    if embs.shape[0] < 2:
        return 0.0
    return _cosine_sim_vec(embs[0], embs[1])


def sentence_on_topic_ratio(topic_content: str, transcript: str, threshold: float) -> Tuple[float, List[Dict[str, Any]]]:
    sents = split_sentences(transcript)
    if not sents:
        return 0.0, []

    embs = embed_texts([topic_content] + sents)
    topic_vec = embs[0]
    sent_vecs = embs[1:]

    sims = _cosine_sim_matrix(topic_vec, sent_vecs)

    per = []
    on = 0
    for s, v in zip(sents, sims):
        simv = float(v)
        ok = simv >= threshold
        on += int(ok)
        per.append({"sentence": s, "similarity": round(simv, 2), "on_topic": ok})

    return round(on / len(sents), 2), per


def semantic_coverage(topic_phrases: List[str], response_phrases: List[str], match_threshold: float = 0.60):
    topic_phrases = [p for p in (topic_phrases or []) if p]
    response_phrases = [p for p in (response_phrases or []) if p]
    if not topic_phrases or not response_phrases:
        return 0.0, [], topic_phrases[:10]

    embs = embed_texts(topic_phrases + response_phrases)
    te = embs[: len(topic_phrases)]
    re_ = embs[len(topic_phrases) :]

    te = te / (np.linalg.norm(te, axis=1, keepdims=True) + 1e-12)
    re_ = re_ / (np.linalg.norm(re_, axis=1, keepdims=True) + 1e-12)

    sims = np.matmul(te, re_.T)  # (T,R)
    best = sims.max(axis=1)

    matched = [p for p, s in zip(topic_phrases, best) if float(s) >= match_threshold]
    missing = [p for p, s in zip(topic_phrases, best) if float(s) < match_threshold]

    cov = len(matched) / max(len(topic_phrases), 1)
    return round(cov, 2), matched[:10], missing[:10]


def relevance_label(score: float) -> str:
    if score >= 0.82:
        return "Highly relevant"
    elif score >= 0.68:
        return "Mostly relevant"
    elif score >= 0.48:
        return "Partially relevant"
    else:
        return "Off-topic"


def run_stage5(topic_obj: Dict[str, Any], transcript: str) -> Dict[str, Any]:
    topic_raw = topic_obj.get("topic_raw", "") or ""
    topic_content = topic_obj.get("topic_content", topic_raw) or ""
    topic_type = topic_obj.get("topic_type", "general")
    expected_anchors = topic_obj.get("expected_anchors", [])
    topic_keyphrases = topic_obj.get("topic_keyphrases", [])

    transcript = (transcript or "").strip()
    topic_focus = normalize_topic_focus(topic_content)

    resp_phrases = yake_keyphrases(transcript, top_k=12)

    sim_full = semantic_similarity(topic_content, transcript)
    sim_focus = semantic_similarity(topic_focus, transcript)
    sim = round((0.55 * sim_full + 0.45 * sim_focus), 4)

    effective_topic_phrases = topic_keyphrases
    if not effective_topic_phrases or len(effective_topic_phrases) < 3:
        effective_topic_phrases = yake_keyphrases(topic_focus or topic_content, top_k=10)

    cov, matched, missing = semantic_coverage(effective_topic_phrases, resp_phrases, match_threshold=0.60)

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

    anchors = {"score": None, "components": {}, "explanation": "No anchor rubric for this topic type."}
    bonus = 0.0

    relevance = 0.62 * sim + 0.18 * cov + 0.20 * on_ratio + bonus
    if sim >= 0.60:
        relevance = max(relevance, 0.70)

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
        "debug": {
            "topic_focus": topic_focus,
            "sim_full": round(sim_full, 2),
            "sim_focus": round(sim_focus, 2),
            "sentence_threshold_used": sent_th,
            "topic_len_words": topic_len,
            "embeddings_provider": EMBEDDINGS_PROVIDER,
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
        },
    }
"""
Stage 6: Learning Guidance & Trust Layer (General Interaction)
Project: GARGI
Author: Krishna

Inputs:
- topic_text (str)
- transcript (str)
- stage4_results (dict): scores, scoring_trace, evidence, feedback
- stage5_results (dict): relevance metrics + sentence on-topic ratio + keyphrases

Outputs (dict):
- confidence: score/label/explanation + components
- priorities: top improvement actions
- coaching_feedback: short, actionable guidance
- reflection_prompts: questions for self-improvement
- session_summary: compact summary suitable for logging
- (optional) writes session log to sessions/sessions.jsonl
"""

from __future__ import annotations

import os
import json
import math
from datetime import datetime
from typing import Dict, Any, List, Tuple


# -----------------------------
# Helpers
# -----------------------------
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def label_from_score(score: float) -> str:
    if score >= 0.80:
        return "High"
    if score >= 0.55:
        return "Medium"
    return "Low"


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

# -----------------------------
# Topic Structure Hint
# -----------------------------
def topic_structure_hint(topic_type: str, expected_anchors: list[str]) -> str:
    if topic_type == "event":
        return "Use: (1) name the event + place, (2) what happened, (3) your perspective, (4) impact/conclusion."
    if topic_type == "advice":
        return "Use: (1) direct answer, (2) 2–3 steps, (3) one example, (4) short closing."
    if topic_type == "opinion":
        return "Use: (1) your position, (2) 2 reasons, (3) a real example, (4) conclusion."
    if topic_type == "compare":
        return "Use: (1) define both items, (2) similarities, (3) differences, (4) conclusion."
    return "Use: 1-sentence answer → 2 points → 1 example → 1 closing sentence."


# -----------------------------
# Confidence Model (heuristic, explainable)
# -----------------------------
def compute_confidence(topic_text: str, transcript: str, stage4: Dict[str, Any], stage5: Dict[str, Any]) -> Dict[str, Any]:
    """
    Confidence reflects how much we should trust the evaluation signals.
    It is NOT "correctness", but an estimate based on signal strength/consistency.
    """

    # Evidence
    wpm = safe_get(stage4, ["evidence", "wpm"], None)
    pause_ratio = safe_get(stage4, ["evidence", "pause_ratio"], None)
    error_density = safe_get(stage4, ["evidence", "error_density"], None)
    grammar_errors = safe_get(stage4, ["evidence", "grammar_errors"], []) or []
    filler_words = safe_get(stage4, ["evidence", "filler_words"], {}) or {}

    sim = stage5.get("semantic_similarity", None)
    on_topic_ratio = stage5.get("on_topic_sentence_ratio", None)

    # Transcript quality proxies
    word_count = len((transcript or "").split())
    char_count = len((transcript or "").strip())

    # Component 1: length adequacy (too short => low confidence)
    # Typical 30–60 sec response: 60–180 words. We prefer >= 60.
    length_score = clamp((word_count - 30) / 90)  # 30->0, 120->1
    length_score = round(length_score, 2)

    # Component 2: signal consistency
    # If stage5 similarity and on-topic ratio exist and broadly agree, confidence increases.
    consistency = 0.5
    if isinstance(sim, (int, float)) and isinstance(on_topic_ratio, (int, float)):
        # If both are high or both are low, consistency is higher; if they conflict, lower.
        # conflict example: sim high but on-topic very low
        diff = abs(float(sim) - float(on_topic_ratio))
        consistency = clamp(1.0 - diff)  # diff 0 => 1, diff 1 => 0
    consistency = round(consistency, 2)

    # Component 3: grammar tool stability
    # If there are some errors but not extreme and density exists, stable.
    grammar_score = 0.7
    if error_density is None:
        grammar_score = 0.5
    else:
        # very high density can indicate transcription noise or low language quality => still stable, but less reliable
        grammar_score = clamp(1.0 - (float(error_density) / 12.0))  # density 0 -> 1, 12 -> 0
    grammar_score = round(grammar_score, 2)

    # Component 4: audio/fluency plausibility proxies
    # If WPM exists and within plausible range, and pause ratio not extreme => more confident.
    fluency_score = 0.6
    if isinstance(wpm, (int, float)):
        wpmf = float(wpm)
        if 90 <= wpmf <= 190:
            fluency_score = 0.9
        elif 70 <= wpmf <= 220:
            fluency_score = 0.75
        else:
            fluency_score = 0.55
    if isinstance(pause_ratio, (int, float)):
        pr = float(pause_ratio)
        if pr > 0.45:
            fluency_score = min(fluency_score, 0.55)
    fluency_score = round(fluency_score, 2)

    # Weighted confidence (simple, explainable)
    # Length matters a lot for trust; consistency also important.
    conf = (
        0.35 * length_score +
        0.25 * consistency +
        0.20 * grammar_score +
        0.20 * fluency_score
    )
    conf = round(clamp(conf), 2)

    explanation_parts = []
    if word_count < 45:
        explanation_parts.append("Short response; some metrics are less reliable.")
    else:
        explanation_parts.append("Adequate response length for stable analysis.")

    if isinstance(sim, (int, float)) and isinstance(on_topic_ratio, (int, float)):
        explanation_parts.append(f"Topic signals consistency: {consistency} (similarity vs sentence-level ratio).")
    else:
        explanation_parts.append("Topic signals partially available; confidence reduced slightly.")

    explanation_parts.append(f"Length={length_score}, GrammarStability={grammar_score}, FluencyPlausibility={fluency_score}.")

    return {
        "confidence_score": conf,
        "confidence_label": label_from_score(conf),
        "confidence_explanation": " ".join(explanation_parts),
        "components": {
            "length_score": length_score,
            "consistency_score": consistency,
            "grammar_stability": grammar_score,
            "fluency_plausibility": fluency_score,
            "word_count": word_count,
            "char_count": char_count,
        }
    }


# -----------------------------
# Priority selection (Top-3)
# -----------------------------
def extract_penalty_impacts(scoring_trace: Dict[str, Any]) -> List[Tuple[str, str, int]]:
    """
    Returns a list of (area, penalty_name, penalty_value) sorted by severity.
    penalty_value is negative in the trace.
    """
    impacts = []
    for area, detail in (scoring_trace or {}).items():
        penalties = detail.get("penalties", []) if isinstance(detail, dict) else []
        for name, val in penalties:
            impacts.append((area, name, int(val)))
    # Most negative first
    impacts.sort(key=lambda x: x[2])
    return impacts


def generate_priority_actions(stage4: Dict[str, Any], stage5: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert penalties + relevance signals into top improvement priorities.
    """
    priorities: List[Dict[str, Any]] = []

    scores = stage4.get("scores", {}) or {}
    trace = stage4.get("scoring_trace", {}) or {}
    evidence = stage4.get("evidence", {}) or {}

    relevance = float(stage5.get("relevance_score", 0.0) or 0.0)
    label = stage5.get("label", "N/A")
    on_topic_ratio = stage5.get("on_topic_sentence_ratio", None)

    impacts = extract_penalty_impacts(trace)

    # 1) Relevance priority (if weak or if sentence ratio low)
    if label in ("Off-topic", "Partially relevant") or (isinstance(on_topic_ratio, (int, float)) and float(on_topic_ratio) < 0.50):
        topic_type = stage5.get("topic_type", "general")
        expected_anchors = stage5.get("expected_anchors", [])
        action_text = topic_structure_hint(topic_type, expected_anchors)

        priorities.append({
            "area": "Topic alignment",
            "severity": "High" if label == "Off-topic" else "Medium",
            "reason": f"Relevance is {relevance} ({label})." + (f" On-topic ratio: {on_topic_ratio}." if on_topic_ratio is not None else ""),
            "action": action_text
        })

    # 2) Penalty-driven priorities (fluency, fillers, grammar)
    for area, penalty_name, penalty_val in impacts:
        if len(priorities) >= 3:
            break

        if area == "fluency" and penalty_name in ("high_wpm", "low_wpm"):
            wpm = evidence.get("wpm")
            priorities.append({
                "area": "Fluency (pace)",
                "severity": "Medium",
                "reason": f"Pace penalty applied ({penalty_name}). WPM={wpm}.",
                "action": "Aim for a steady pace. Use short pauses at sentence boundaries, not mid-phrase."
            })

        elif area == "fluency" and "pause_ratio" in penalty_name:
            pr = evidence.get("pause_ratio")
            priorities.append({
                "area": "Fluency (pausing)",
                "severity": "Medium",
                "reason": f"Pause penalty applied ({penalty_name}). Pause ratio={pr}.",
                "action": "Reduce long silences by planning your next sentence as you finish the current one."
            })

        elif area == "fillers":
            fillers = evidence.get("filler_words", {} )
            top = sorted(fillers.items(), key=lambda x: x[1], reverse=True)[:3] if isinstance(fillers, dict) else []
            priorities.append({
                "area": "Fillers",
                "severity": "Medium",
                "reason": f"Filler penalty applied ({penalty_name}). Top fillers: {top}.",
                "action": "Replace filler words with a silent pause. Practice speaking in short, complete sentences."
            })

        elif area == "grammar":
            errs = evidence.get("grammar_errors", [])
            top_rules = {}
            if isinstance(errs, list):
                for e in errs:
                    rid = (e or {}).get("rule", "UNKNOWN")
                    top_rules[rid] = top_rules.get(rid, 0) + 1
            top_rules_sorted = sorted(top_rules.items(), key=lambda x: x[1], reverse=True)[:3]
            priorities.append({
                "area": "Grammar",
                "severity": "Medium",
                "reason": f"Grammar penalty applied ({penalty_name}). Common rules: {top_rules_sorted}.",
                "action": "Review the most frequent grammar patterns and rewrite 3–5 sentences from your transcript using the suggestions."
            })

    # 3) If no penalties and relevance is fine: suggest structure upgrade
    if not priorities:
        priorities.append({
            "area": "Structure",
            "severity": "Low",
            "reason": "Quality signals are strong; improve clarity through structure.",
            "action": "Use a simple structure: 1-sentence answer → 2 supporting points → 1 example → 1 closing sentence."
        })

    # Keep at most 3
    return priorities[:3]



# -----------------------------
# Coaching feedback templates
# -----------------------------
def generate_coaching_feedback(stage4: Dict[str, Any], stage5: Dict[str, Any]) -> List[str]:
    scores = stage4.get("scores", {}) or {}
    evidence = stage4.get("evidence", {}) or {}

    overall = scores.get("overall", None)
    fluency = scores.get("fluency", None)
    grammar = scores.get("grammar", None)
    fillers = scores.get("fillers", None)

    wpm = evidence.get("wpm", None)
    pause_ratio = evidence.get("pause_ratio", None)
    filler_words = evidence.get("filler_words", {}) or {}

    relevance = stage5.get("relevance_score", None)
    on_topic_ratio = stage5.get("on_topic_sentence_ratio", None)
    response_keyphrases = stage5.get("response_keyphrases", []) or []

    out: List[str] = []

    if isinstance(overall, (int, float)):
        out.append(f"Overall, your communication quality is strong ({overall}/10).")

    if isinstance(relevance, (int, float)):
        if relevance >= 0.80:
            out.append("Your response stayed well-aligned with the topic.")
        elif relevance >= 0.70:
            out.append("Your response was mostly aligned with the topic; you can tighten focus by reducing tangents.")
        else:
            out.append("Your response drifted from the topic; start with a one-sentence answer and keep each point tied to the prompt.")

    if isinstance(on_topic_ratio, (int, float)):
        out.append(f"On-topic content ratio (sentence-level): {on_topic_ratio}. Aim for 0.60+ for general interaction tasks.")

    if isinstance(wpm, (int, float)):
        out.append(f"Speaking rate: {wpm} WPM. For clear general interaction, a typical target is ~120–170 WPM.")

    if isinstance(pause_ratio, (int, float)):
        out.append(f"Pausing: ratio={pause_ratio}. Short pauses between sentences are good; long silences reduce clarity.")

    if isinstance(filler_words, dict) and filler_words:
        top = sorted(filler_words.items(), key=lambda x: x[1], reverse=True)[:3]
        out.append(f"Top filler words: {top}. Replace them with a brief silent pause.")

    if response_keyphrases:
        out.append(f"Key themes you discussed: {', '.join(response_keyphrases[:6])}.")

    return out


# -----------------------------
# Reflection prompts (general interaction)
# -----------------------------
def generate_reflection_prompts(stage5: Dict[str, Any]) -> List[str]:
    label = stage5.get("label", "N/A")
    on_topic_ratio = stage5.get("on_topic_sentence_ratio", None)

    prompts = [
        "Did you answer the prompt directly in your first 1–2 sentences?",
        "What was your main point, stated in one sentence?",
        "Which sentence best supports your main point?",
        "What is one sentence you would remove to make your response more focused?",
        "If you spoke again, what example would you add to make your point clearer?",
    ]

    if label in ("Off-topic", "Partially relevant"):
        prompts.insert(0, "Where did you begin to drift off-topic, and what was the trigger (example, memory, unrelated detail)?")

    if isinstance(on_topic_ratio, (int, float)) and float(on_topic_ratio) < 0.50:
        prompts.insert(1, "Try rewriting your response outline: 1 direct answer + 2 supporting points + 1 example + 1 closing sentence.")

    return prompts[:6]


# -----------------------------
# Session logging
# -----------------------------
def write_session_log(session_obj: Dict[str, Any], out_dir: str = "sessions", filename: str = "sessions.jsonl") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(session_obj, ensure_ascii=False) + "\n")
    return path


# -----------------------------
# Public API
# -----------------------------
def run_stage6(topic_text: str, transcript: str, stage4_results: Dict[str, Any], stage5_results: Dict[str, Any],
               save_history: bool = True) -> Dict[str, Any]:
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    confidence = compute_confidence(topic_text, transcript, stage4_results, stage5_results)
    priorities = generate_priority_actions(stage4_results, stage5_results)
    coaching_feedback = generate_coaching_feedback(stage4_results, stage5_results)
    reflection_prompts = generate_reflection_prompts(stage5_results)

    # Compact summary for tracking
    scores = stage4_results.get("scores", {}) or {}
    session_summary = {
        "timestamp_utc": now,
        "topic": topic_text,
        "overall_quality_score": scores.get("overall"),
        "fluency_score": scores.get("fluency"),
        "grammar_score": scores.get("grammar"),
        "fillers_score": scores.get("fillers"),
        "relevance_score": stage5_results.get("relevance_score"),
        "relevance_label": stage5_results.get("label"),
        "on_topic_sentence_ratio": stage5_results.get("on_topic_sentence_ratio"),
        "confidence_score": confidence.get("confidence_score"),
        "confidence_label": confidence.get("confidence_label"),
    }

    log_path = None
    if save_history:
        log_path = write_session_log(session_summary)

    return {
        "confidence": confidence,
        "priorities": priorities,
        "coaching_feedback": coaching_feedback,
        "reflection_prompts": reflection_prompts,
        "session_summary": session_summary,
        "history_log_path": log_path,
    }

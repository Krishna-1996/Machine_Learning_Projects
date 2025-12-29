"""
Stage 4: Scoring, Feedback & Explainability (Robust to LanguageTool fallback)
Project: GARGI
Author: Krishna
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional


# -----------------------------
# Scoring functions
# -----------------------------
def score_fluency(wpm: float, pause_ratio: float) -> Tuple[int, List[Tuple[str, int]], int]:
    base = 10
    penalties: List[Tuple[str, int]] = []

    if wpm < 90:
        penalties.append(("low_wpm", -2))
    elif wpm > 170:
        penalties.append(("high_wpm", -2))

    if pause_ratio > 0.30:
        penalties.append(("high_pause_ratio", -3))
    elif pause_ratio > 0.20:
        penalties.append(("moderate_pause_ratio", -2))

    final = max(base + sum(p[1] for p in penalties), 0)
    return base, penalties, final


def score_fillers(filler_words: Any) -> Tuple[int, List[Tuple[str, int]], int]:
    base = 10
    total_fillers = sum(filler_words.values()) if isinstance(filler_words, dict) else 0
    penalties: List[Tuple[str, int]] = []

    if total_fillers > 6:
        penalties.append(("excessive_fillers", -6))
    elif total_fillers > 3:
        penalties.append(("moderate_fillers", -4))
    elif total_fillers > 0:
        penalties.append(("few_fillers", -2))

    final = max(base + sum(p[1] for p in penalties), 0)
    return base, penalties, final


def score_grammar(error_density: float) -> Tuple[int, List[Tuple[str, int]], int]:
    base = 10
    penalties: List[Tuple[str, int]] = []

    if error_density >= 8:
        penalties.append(("very_high_error_density", -7))
    elif error_density >= 5:
        penalties.append(("high_error_density", -5))
    elif error_density >= 2:
        penalties.append(("moderate_error_density", -3))

    final = max(base + sum(p[1] for p in penalties), 0)
    return base, penalties, final


# -----------------------------
# Helpers to support BOTH schemas
# -----------------------------
def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        return float(str(x))
    except Exception:
        return default


def _get_fluency_block(stage3_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supports:
      A) nested schema: {"fluency": {"wpm":..., "pause_ratio":..., "filler_words":...}}
      B) flat schema (your current): {"wpm":..., "pause_ratio":..., "filler_words":...}
    """
    flu = stage3_data.get("fluency")
    if isinstance(flu, dict) and flu:
        return flu

    return {
        "wpm": stage3_data.get("wpm", 0.0),
        "pause_ratio": stage3_data.get("pause_ratio", 0.0),
        "filler_words": stage3_data.get("filler_words", {}) or {},
    }


def _get_grammar_block(stage3_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supports:
      A) nested schema: {"grammar": {"total_errors":..., "error_density":..., "errors":[...], "warning":...}}
      B) flat schema (your current): {"grammar_raw": {...}}, OR sometimes list, OR {"matches":[...]}
    """
    g = stage3_data.get("grammar")
    if isinstance(g, dict) and g:
        return g

    # Preferred fallback: grammar_raw dict from LanguageTool-style output
    raw = stage3_data.get("grammar_raw")

    # If grammar_raw is a dict that already contains needed fields, return it
    if isinstance(raw, dict):
        # Some grammar analyzers store errors in "matches" instead of "errors"
        if "errors" not in raw and isinstance(raw.get("matches"), list):
            # Map to expected key for downstream processing
            raw = dict(raw)
            raw["errors"] = raw.get("matches", [])
        return raw

    # If grammar_raw is a list of error objects
    if isinstance(raw, list):
        return {
            "total_errors": len(raw),
            "error_density": stage3_data.get("error_density", 0.0) or 0.0,
            "errors": raw,
            "warning": None,
        }

    # If nothing else, return empty grammar block
    return {
        "total_errors": stage3_data.get("grammar_errors", 0) or 0,
        "error_density": stage3_data.get("error_density", 0.0) or 0.0,
        "errors": [],
        "warning": None,
    }


def _summarize_grammar_rules(grammar_errors: Any, top_k: int = 3) -> List[Tuple[str, int]]:
    rule_counts: Dict[str, int] = {}

    if not isinstance(grammar_errors, list):
        return []

    for err in grammar_errors:
        if not isinstance(err, dict):
            continue
        rid = err.get("rule") or err.get("ruleId") or "UNKNOWN_RULE"
        rule_counts[rid] = rule_counts.get(rid, 0) + 1

    ranked = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# -----------------------------
# Feedback generation
# -----------------------------
def generate_feedback(stage3_data: Dict[str, Any]) -> List[str]:
    """
    Human-readable feedback, consistent with scoring policy.
    Robust to missing keys and multiple grammar output shapes.
    """
    feedback: List[str] = []

    fluency = _get_fluency_block(stage3_data)
    grammar = _get_grammar_block(stage3_data)

    wpm = _as_float(fluency.get("wpm", 0.0), 0.0)
    pause_ratio = _as_float(fluency.get("pause_ratio", 0.0), 0.0)
    filler_words = fluency.get("filler_words", {}) or {}

    total_errors = int(_as_float(grammar.get("total_errors", 0), 0.0))
    error_density = _as_float(grammar.get("error_density", 0.0), 0.0)
    grammar_errors = grammar.get("errors", []) or []
    warning = grammar.get("warning", None)

    # ---- Fluency feedback
    if wpm < 100:
        feedback.append("Your speaking pace was slow. Aim for a steady rhythm.")
    elif wpm > 160:
        feedback.append("Your speaking pace was fast. Slowing down may improve clarity.")
    else:
        feedback.append("Your speaking pace was appropriate.")

    if pause_ratio > 0.25:
        feedback.append("You paused frequently. Try reducing long silences.")

    # ---- Filler feedback (show top offenders)
    if isinstance(filler_words, dict) and filler_words:
        top_fillers = sorted(filler_words.items(), key=lambda x: x[1], reverse=True)[:3]
        for word, count in top_fillers:
            try:
                if int(count) >= 2:
                    feedback.append(
                        f"You used the filler word '{word}' {count} times. Consider replacing it with a silent pause."
                    )
            except Exception:
                continue

    # ---- Grammar feedback
    if warning:
        feedback.append("Grammar engine was unavailable; grammar feedback may be incomplete.")

    if total_errors == 0:
        feedback.append("No grammar issues were detected.")
    else:
        feedback.append(
            f"{total_errors} grammar issue(s) detected; overall accuracy remains high "
            f"({round(error_density, 2)} errors per 100 words)."
        )
        top_rules = _summarize_grammar_rules(grammar_errors, top_k=3)
        if top_rules:
            rule_text = ", ".join([f"{r}×{c}" for r, c in top_rules])
            feedback.append(f"Most frequent grammar rule(s): {rule_text}.")

    return feedback


# -----------------------------
# Main Stage 4 runner
# -----------------------------
def run_stage4(stage3_data: Dict[str, Any]) -> Dict[str, Any]:
    fluency = _get_fluency_block(stage3_data)
    grammar = _get_grammar_block(stage3_data)

    wpm = _as_float(fluency.get("wpm", 0.0), 0.0)
    pause_ratio = _as_float(fluency.get("pause_ratio", 0.0), 0.0)
    filler_words = fluency.get("filler_words", {}) or {}

    error_density = _as_float(grammar.get("error_density", 0.0), 0.0)
    grammar_errors = grammar.get("errors", []) or []

    f_base, f_penalties, f_final = score_fluency(wpm, pause_ratio)
    fl_base, fl_penalties, fl_final = score_fillers(filler_words)
    g_base, g_penalties, g_final = score_grammar(error_density)

    overall = round(
        0.4 * f_final +
        0.3 * g_final +
        0.3 * fl_final,
        1
    )

    # Important: Ensure feedback reads the same schema (nested) consistently
    feedback = generate_feedback({
        "fluency": fluency,
        "grammar": grammar,
    })

    return {
        "scores": {
            "fluency": f_final,
            "grammar": g_final,
            "fillers": fl_final,
            "overall": overall
        },
        "scoring_trace": {
            "fluency": {"base": f_base, "penalties": f_penalties, "final": f_final},
            "grammar": {"base": g_base, "penalties": g_penalties, "final": g_final},
            "fillers": {"base": fl_base, "penalties": fl_penalties, "final": fl_final}
        },
        "evidence": {
            "wpm": wpm,
            "pause_ratio": pause_ratio,
            "filler_words": filler_words,
            "grammar_errors": grammar_errors,
            "error_density": error_density,
            "grammar_warning": grammar.get("warning", None),
        },
        "feedback": feedback
    }

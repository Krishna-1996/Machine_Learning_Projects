"""
Stage 4: Scoring, Feedback & Explainability (Robust to LanguageTool fallback)
Project: GARGI
Author: Krishna
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def score_fluency(wpm: float, pause_ratio: float):
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


def score_fillers(filler_words: Any):
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


def score_grammar(error_density: float):
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


def _summarize_grammar_rules(grammar_errors: List[Dict[str, Any]], top_k: int = 3):
    rule_counts: Dict[str, int] = {}
    for err in grammar_errors or []:
        rid = (err or {}).get("rule", "UNKNOWN_RULE")
        rule_counts[rid] = rule_counts.get(rid, 0) + 1

    ranked = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def _extract_grammar_parts(grammar_raw: Any):
    """
    Normalize grammar_raw into:
      total_errors: int
      error_density: float
      errors: list[dict]
      warning: Optional[str]
    Supports:
      - dict with keys (total_errors, error_density, errors, warning)
      - LanguageTool-like dict with 'matches' list
      - list of errors
    """
    total_errors = 0
    error_density = 0.0
    errors: List[Dict[str, Any]] = []
    warning = None

    if grammar_raw is None:
        return total_errors, error_density, errors, warning

    if isinstance(grammar_raw, dict):
        if "total_errors" in grammar_raw:
            total_errors = int(grammar_raw.get("total_errors") or 0)
        if "error_density" in grammar_raw:
            try:
                error_density = float(grammar_raw.get("error_density") or 0.0)
            except Exception:
                error_density = 0.0
        if "errors" in grammar_raw and isinstance(grammar_raw.get("errors"), list):
            errors = grammar_raw.get("errors") or []
        if "warning" in grammar_raw:
            warning = grammar_raw.get("warning")

        # LanguageTool-style fallback
        matches = grammar_raw.get("matches")
        if isinstance(matches, list) and not errors:
            # Convert LT matches into your error shape
            for m in matches:
                errors.append({
                    "rule": (m.get("rule", {}) or {}).get("id", "LT_RULE"),
                    "message": m.get("message", ""),
                    "suggestions": [r.get("value") for r in (m.get("replacements") or []) if isinstance(r, dict)]
                })
            total_errors = max(total_errors, len(matches))

        return total_errors, error_density, errors, warning

    if isinstance(grammar_raw, list):
        # already list of errors
        errors = grammar_raw
        total_errors = len(errors)
        # error_density unknown here; keep 0 unless you compute it earlier
        return total_errors, error_density, errors, warning

    return total_errors, error_density, errors, warning


def generate_feedback(stage3_data: Dict[str, Any]) -> List[str]:
    """
    Human-readable feedback, consistent with scoring policy.
    Uses your current stage3 schema keys.
    """
    feedback: List[str] = []

    wpm = float(stage3_data.get("wpm", 0) or 0)
    pause_ratio = float(stage3_data.get("pause_ratio", 0) or 0)
    filler_words = stage3_data.get("filler_words", {}) or {}

    grammar_raw = stage3_data.get("grammar_raw")
    total_errors, error_density, grammar_errors, warning = _extract_grammar_parts(grammar_raw)

    # ---- Fluency feedback
    if wpm < 100:
        feedback.append("Your speaking pace was slow. Aim for a steady rhythm.")
    elif wpm > 160:
        feedback.append("Your speaking pace was fast. Slowing down may improve clarity.")
    else:
        feedback.append("Your speaking pace was appropriate.")

    if pause_ratio > 0.25:
        feedback.append("You paused frequently. Try reducing long silences.")

    # ---- Filler feedback
    if isinstance(filler_words, dict) and filler_words:
        top_fillers = sorted(filler_words.items(), key=lambda x: x[1], reverse=True)[:3]
        for word, count in top_fillers:
            if int(count) >= 2:
                feedback.append(
                    f"You used the filler word '{word}' {count} times. Consider replacing it with a silent pause."
                )

    # ---- Grammar feedback
    if warning:
        feedback.append("Grammar engine was unavailable; grammar feedback may be incomplete.")

    if total_errors == 0:
        feedback.append("No grammar issues were detected.")
    else:
        feedback.append(
            f"{total_errors} grammar issue(s) detected; overall accuracy remains high "
            f"({error_density} errors per 100 words)."
        )
        top_rules = _summarize_grammar_rules(grammar_errors, top_k=3)
        if top_rules:
            # ASCII-safe "x" instead of "×"
            rule_text = ", ".join([f"{r} x{c}" for r, c in top_rules])
            feedback.append(f"Most frequent grammar rule(s): {rule_text}.")

    return feedback


def run_stage4(stage3_data: Dict[str, Any]) -> Dict[str, Any]:
    wpm = float(stage3_data.get("wpm", 0) or 0)
    pause_ratio = float(stage3_data.get("pause_ratio", 0) or 0)
    filler_words = stage3_data.get("filler_words", {}) or {}

    grammar_raw = stage3_data.get("grammar_raw")
    total_errors, error_density, grammar_errors, warning = _extract_grammar_parts(grammar_raw)

    f_base, f_penalties, f_final = score_fluency(wpm, pause_ratio)
    fl_base, fl_penalties, fl_final = score_fillers(filler_words)
    g_base, g_penalties, g_final = score_grammar(float(error_density or 0.0))

    overall = round(
        0.4 * f_final +
        0.3 * g_final +
        0.3 * fl_final,
        1
    )

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
            "error_density": float(error_density or 0.0),
            "grammar_warning": warning,
        },
        "feedback": generate_feedback(stage3_data)
    }

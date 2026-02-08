from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional


# -------------------------------
# Robust extractors (support old + new stage3 schemas)
# -------------------------------

def _extract_fluency(stage3: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supports BOTH schemas:

    NEW schema (flat):
      stage3["wpm"], stage3["pause_ratio"], stage3["filler_words"], stage3["duration_sec"], stage3["word_count"]

    OLD schema (nested):
      stage3["fluency"]["wpm"], stage3["fluency"]["pause_ratio"], stage3["fluency"]["filler_words"]
      OR stage3["fillers"] (dict of counts)
      OR stage3["fluency"]["filler_count"] (int)
      word_count may be in stage3["lexical"]["word_count"]
    """
    flu = stage3.get("fluency") if isinstance(stage3.get("fluency"), dict) else {}

    # WPM
    wpm = stage3.get("wpm")
    if wpm is None:
        wpm = flu.get("wpm", 0.0)

    # Pause ratio
    pause_ratio = stage3.get("pause_ratio")
    if pause_ratio is None:
        pause_ratio = flu.get("pause_ratio", 0.0)

    # Duration / word_count
    duration_sec = stage3.get("duration_sec")
       if duration_sec is None:
        duration_sec = flu.get("duration_sec")

    word_count = stage3.get("word_count")
    if word_count is None:
        word_count = flu.get("word_count")
    if word_count is None:
        lex = stage3.get("lexical") if isinstance(stage3.get("lexical"), dict) else {}
        word_count = lex.get("word_count")

    # Filler words (dict)
    filler_words = stage3.get("filler_words")
    if not isinstance(filler_words, dict):
        filler_words = flu.get("filler_words")
    if not isinstance(filler_words, dict):
        filler_words = stage3.get("fillers")  # old schema
    if not isinstance(filler_words, dict):
        filler_words = {}

    # If old schema only has filler_count, convert to a pseudo-dict so scoring still works
    if not filler_words:
        filler_count = flu.get("filler_count")
        if isinstance(filler_count, (int, float)) and filler_count > 0:
            filler_words = {"(fillers_total)": int(filler_count)}

    return {
        "wpm": float(wpm or 0.0),
        "pause_ratio": float(pause_ratio or 0.0),
        "duration_sec": duration_sec,
        "word_count": word_count,
        "filler_words": filler_words,
    }


def _extract_grammar_raw(stage3: Dict[str, Any]) -> Any:
    """
    NEW schema: stage3["grammar_raw"]
    OLD schema: stage3["grammar"]
    """
    if "grammar_raw" in stage3:
        return stage3.get("grammar_raw")
    return stage3.get("grammar")


# -------------------------------
# Existing scoring functions (keep your current logic)
# -------------------------------

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


def _extract_grammar_parts(grammar_raw: Any):
    # Keep your existing implementation as-is.
    # (Use the one already in your file. Not duplicating here to avoid conflicts.)
    raise NotImplementedError("KEEP YOUR EXISTING _extract_grammar_parts IMPLEMENTATION")


def generate_feedback(stage3_data: Dict[str, Any]) -> List[str]:
    """
    Updated to use robust extractor so feedback never says WPM=0 unless it truly is.
    """
    feedback: List[str] = []

    flu = _extract_fluency(stage3_data)
    wpm = float(flu["wpm"])
    pause_ratio = float(flu["pause_ratio"])
    filler_words = flu["filler_words"]

    grammar_raw = _extract_grammar_raw(stage3_data)
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
            if int(count) >= 2 and word != "(fillers_total)":
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

    return feedback


def run_stage4(stage3_data: Dict[str, Any]) -> Dict[str, Any]:
    flu = _extract_fluency(stage3_data)
    wpm = float(flu["wpm"])
    pause_ratio = float(flu["pause_ratio"])
    filler_words = flu["filler_words"]

    grammar_raw = _extract_grammar_raw(stage3_data)
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
            "duration_sec": flu.get("duration_sec"),
            "word_count": flu.get("word_count"),
        },
        "feedback": generate_feedback(stage3_data)
    }
"""
Stage 4: Scoring, Feedback & Explainability (XAI-style)
Project: GARGI
Author: Krishna
"""

# -------------------------------
# Scoring Functions (Pure Rules)
# -------------------------------

def score_fluency(wpm, pause_ratio):
    base = 10
    penalties = []

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


def score_fillers(filler_words):
    base = 10
    total_fillers = sum(filler_words.values())
    penalties = []

    if total_fillers > 6:
        penalties.append(("excessive_fillers", -6))
    elif total_fillers > 3:
        penalties.append(("moderate_fillers", -4))
    elif total_fillers > 0:
        penalties.append(("few_fillers", -2))

    final = max(base + sum(p[1] for p in penalties), 0)
    return base, penalties, final


def score_grammar(error_density):
    base = 10
    penalties = []

    if error_density >= 8:
        penalties.append(("very_high_error_density", -7))
    elif error_density >= 5:
        penalties.append(("high_error_density", -5))
    elif error_density >= 2:
        penalties.append(("moderate_error_density", -3))

    final = max(base + sum(p[1] for p in penalties), 0)
    return base, penalties, final


# -------------------------------
# Feedback Generator
# -------------------------------

def generate_feedback(stage3_data):
    feedback = []

    fluency = stage3_data["fluency"]
    grammar = stage3_data["grammar"]

    # Fluency
    if fluency["wpm"] < 100:
        feedback.append("Your speaking pace was slow. Aim for a steady rhythm.")
    elif fluency["wpm"] > 160:
        feedback.append("Your speaking pace was fast. Slowing down may improve clarity.")
    else:
        feedback.append("Your speaking pace was appropriate.")

    if fluency["pause_ratio"] > 0.25:
        feedback.append("You paused frequently. Try reducing long silences.")

    # Fillers
    for word, count in fluency["filler_words"].items():
        if count > 2:
            feedback.append(
                f"You used the filler word '{word}' {count} times. Replace it with silent pauses."
            )

    # Grammar
    if grammar["total_errors"] > 0:
        feedback.append(
            f"{grammar['total_errors']} grammar issues were detected. Review sentence construction."
        )
    else:
        feedback.append("Your grammar usage was strong.")

    return feedback


# -------------------------------
# Stage 4 Orchestrator
# -------------------------------

def run_stage4(stage3_data):
    fluency = stage3_data["fluency"]
    grammar = stage3_data["grammar"]

    # --- Fluency ---
    f_base, f_penalties, f_final = score_fluency(
        fluency["wpm"],
        fluency["pause_ratio"]
    )

    # --- Fillers ---
    fl_base, fl_penalties, fl_final = score_fillers(
        fluency["filler_words"]
    )

    # --- Grammar ---
    g_base, g_penalties, g_final = score_grammar(
        grammar["error_density"]
    )

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
            "fluency": {
                "base": f_base,
                "penalties": f_penalties,
                "final": f_final
            },
            "grammar": {
                "base": g_base,
                "penalties": g_penalties,
                "final": g_final
            },
            "fillers": {
                "base": fl_base,
                "penalties": fl_penalties,
                "final": fl_final
            }
        },

        "evidence": {
            "wpm": fluency["wpm"],
            "pause_ratio": fluency["pause_ratio"],
            "filler_words": fluency["filler_words"],
            "grammar_errors": grammar["errors"],
            "error_density": grammar["error_density"]
        },

        "feedback": generate_feedback(stage3_data)
    }

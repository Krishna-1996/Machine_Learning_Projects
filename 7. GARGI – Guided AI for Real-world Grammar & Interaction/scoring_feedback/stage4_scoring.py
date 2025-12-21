"""
Stage 4: Scoring & Feedback Engine
Project: GARGI
Author: Krishna
"""

def score_fluency(wpm, pause_ratio):
    score = 10

    if wpm < 90:
        score -= 2
    elif wpm > 170:
        score -= 2

    if pause_ratio > 0.30:
        score -= 3
    elif pause_ratio > 0.20:
        score -= 2

    return max(score, 0)


def score_fillers(filler_counts):
    total_fillers = sum(filler_counts.values())

    if total_fillers == 0:
        return 10
    elif total_fillers <= 3:
        return 8
    elif total_fillers <= 6:
        return 6
    else:
        return 4


def score_grammar(error_density):
    if error_density < 2:
        return 9
    elif error_density < 5:
        return 7
    elif error_density < 8:
        return 5
    else:
        return 3


def generate_feedback(stage3_data):
    fluency = stage3_data["fluency"]
    grammar = stage3_data["grammar"]

    feedback = []

    # ---- Fluency Feedback ----
    wpm = fluency["wpm"]
    pause_ratio = fluency["pause_ratio"]

    if wpm < 100:
        feedback.append("You spoke a bit slowly. Try maintaining a steady pace.")
    elif wpm > 160:
        feedback.append("You spoke quite fast. Slowing down may improve clarity.")
    else:
        feedback.append("Your speaking pace was appropriate.")

    if pause_ratio > 0.25:
        feedback.append("There were frequent long pauses. Try speaking more smoothly.")

    # ---- Filler Feedback ----
    for word, count in fluency["filler_words"].items():
        if count > 2:
            feedback.append(
                f"You used the filler word '{word}' {count} times. Try pausing silently instead."
            )

    # ---- Grammar Feedback ----
    if grammar["total_errors"] > 0:
        feedback.append(
            f"{grammar['total_errors']} grammar issues were detected. Review sentence structure and verb tenses."
        )
    else:
        feedback.append("Your grammar usage was strong.")

    return feedback


def run_stage4(stage3_data):
    fluency = stage3_data["fluency"]
    grammar = stage3_data["grammar"]

    fluency_score = score_fluency(
        fluency["wpm"],
        fluency["pause_ratio"]
    )

    filler_score = score_fillers(
        fluency["filler_words"]
    )

    grammar_score = score_grammar(
        grammar["error_density"]
    )

    final_score = round(
        0.4 * fluency_score +
        0.3 * grammar_score +
        0.3 * filler_score,
        1
    )

    feedback = generate_feedback(stage3_data)

    return {
        "scores": {
            "fluency": fluency_score,
            "fillers": filler_score,
            "grammar": grammar_score,
            "overall": final_score
        },
        "feedback": feedback
    }

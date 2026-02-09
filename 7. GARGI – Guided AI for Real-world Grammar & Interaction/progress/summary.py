def generate_summary(level, total_score, trends):
    if not trends:
        return "This is the learner’s first recorded attempt."

    improving = [m for m, d in trends.items() if d > 0]
    declining = [m for m, d in trends.items() if d < 0]

    summary = []

    if improving:
        summary.append(
            f"Improvement observed in {', '.join(improving)}."
        )

    if declining:
        summary.append(
            f"Needs attention in {', '.join(declining)}."
        )

    if not summary:
        summary.append("Performance is stable across attempts.")

    summary.append(f"Overall performance level: {level}, score {total_score}.")

    return " ".join(summary)

def generate_paragraph(level, score):
    if level == "kids":
        return "Great job speaking confidently! Keep expressing your ideas clearly."

    if level == "beginner":
        return "You are speaking clearly and building fluency. Keep practicing to gain confidence."

    if level == "intermediate":
        return "Your speech shows good structure and flow. Focus on reducing small grammatical issues."

    return "Your response is professional and mostly interview-ready. Minor refinements will improve impact."


def explainability_output(level, score, breakdown):
    paragraph = generate_paragraph(level, score)

    if level in ["kids", "beginner"]:
        return {"paragraph": paragraph}

    if level == "intermediate":
        return {
            "paragraph": paragraph,
            "breakdown": breakdown
        }

    return {
        "paragraph": paragraph,
        "breakdown": breakdown,
        "table": breakdown
    }

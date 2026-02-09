LEVEL_TONE = {
    "kids": "encouraging",
    "beginner": "supportive",
    "intermediate": "neutral",
    "advanced": "direct"
}

def apply_tone(text, level):
    tone = LEVEL_TONE.get(level, "neutral")

    if tone == "encouraging":
        return f"Good try! {text}"
    if tone == "supportive":
        return f"{text} Keep practicing."
    if tone == "direct":
        return text

    return text

import re

FILLERS = ["um", "uh", "erm", "like"]

def fluency_score(transcript: str, duration: float):
    words = transcript.split()
    word_count = len(words)

    wpm = word_count / (duration / 60)

    filler_count = sum(
        len(re.findall(rf"\b{f}\b", transcript.lower()))
        for f in FILLERS
    )

    filler_penalty = min(0.4, filler_count * 0.05)
    speed_penalty = 0.0

    if wpm < 80:
        speed_penalty = 0.15
    elif wpm > 180:
        speed_penalty = 0.10

    score = max(0.0, 1.0 - filler_penalty - speed_penalty)

    explanation = (
        f"Speech rate: {int(wpm)} WPM. "
        f"{filler_count} filler words detected."
    )

    return round(score, 2), explanation

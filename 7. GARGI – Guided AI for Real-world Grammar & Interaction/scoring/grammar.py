import re

COMMON_ERRORS = [
    r"\bi is\b",
    r"\bhe go\b",
    r"\bshe go\b",
    r"\bthey goes\b",
]

def grammar_score(transcript: str):
    errors = sum(
        len(re.findall(pattern, transcript.lower()))
        for pattern in COMMON_ERRORS
    )

    penalty = min(0.6, errors * 0.15)
    score = max(0.0, 1.0 - penalty)

    explanation = (
        f"{errors} common grammatical patterns flagged."
        if errors > 0 else
        "No obvious grammatical patterns flagged."
    )

    return round(score, 2), explanation

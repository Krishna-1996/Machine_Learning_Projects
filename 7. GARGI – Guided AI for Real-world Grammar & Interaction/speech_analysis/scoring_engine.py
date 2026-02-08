LEVEL_WEIGHTS = {
    "kids": {"confidence": 0.6, "topic": 0.4},
    "beginner": {"fluency": 0.4, "topic": 0.2, "confidence": 0.4},
    "intermediate": {"grammar": 0.25, "fluency": 0.25, "topic": 0.25, "confidence": 0.25},
    "advanced": {"grammar": 0.2, "fluency": 0.2, "topic": 0.2, "confidence": 0.2, "interview": 0.2}
}

def score_by_level(level: str, metrics: dict):
    weights = LEVEL_WEIGHTS[level]
    breakdown = {}
    total = 0

    for k, w in weights.items():
        score = 100 * w
        breakdown[k] = score
        total += score

    return {
        "level": level,
        "score": round(total),
        "breakdown": breakdown,
        "raw_metrics": metrics
    }

from scoring.weights import LEVEL_WEIGHTS

def score_fluency(fluency):
    wpm = fluency["wpm"]
    fillers = sum(fluency["filler_words"].values())

    score = 100

    if wpm < 90 or wpm > 160:
        score -= 20

    score -= fillers * 2
    return max(score, 0)


def score_grammar(grammar):
    return max(100 - grammar["total_errors"] * 5, 0)


def score_confidence(fluency):
    fillers = sum(fluency["filler_words"].values())
    pause_ratio = fluency["pause_ratio"]

    score = 100
    score -= fillers * 3
    score -= int(pause_ratio * 50)
    return max(score, 0)


def score_topic(transcript):
    return 70 if len(transcript.split()) > 30 else 50


def score_interview(fluency, grammar):
    base = score_fluency(fluency)
    penalty = grammar["total_errors"] * 5
    return max(base - penalty, 0)


def compute_final_score(level: str, transcript: str, analysis: dict):
    """
    Sprint 9 rule:
    - Missing analysis signals default to 0.0
    - Never raise KeyError
    - Deterministic scoring
    """

    weights = LEVEL_WEIGHTS[level]

    components = {}
    total_score = 0.0

    for metric, weight in weights.items():
        raw_value = float(analysis.get(metric, 0.0))
        weighted_score = raw_value * weight

        components[metric] = weighted_score
        total_score += weighted_score

    return int(round(total_score * 100)), components

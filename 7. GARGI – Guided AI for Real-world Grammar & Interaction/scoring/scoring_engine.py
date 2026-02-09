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


def compute_final_score(level, transcript, analysis):
    weights = LEVEL_WEIGHTS[level]
    fluency = analysis["fluency"]
    grammar = analysis["grammar"]

    components = {}
    total = 0

    for metric, weight in weights.items():
        if metric == "fluency":
            raw = score_fluency(fluency)
        elif metric == "grammar":
            raw = score_grammar(grammar)
        elif metric == "confidence":
            raw = score_confidence(fluency)
        elif metric == "topic":
            raw = score_topic(transcript)
        elif metric == "interview":
            raw = score_interview(fluency, grammar)
        else:
            raw = 0

        weighted = round(raw * weight, 1)
        components[metric] = weighted
        total += weighted

    return round(total), components

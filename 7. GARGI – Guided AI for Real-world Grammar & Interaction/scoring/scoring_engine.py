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
    # simple placeholder (rule-based)
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
            val = score_fluency(fluency)
        elif metric == "grammar":
            val = score_grammar(grammar)
        elif metric == "confidence":
            val = score_confidence(fluency)
        elif metric == "topic":
            val = score_topic(transcript)
        elif metric == "interview":
            val = score_interview(fluency, grammar)
        else:
            val = 0

        components[metric] = round(val * weight, 1)
        total += components[metric]

    return round(total), components

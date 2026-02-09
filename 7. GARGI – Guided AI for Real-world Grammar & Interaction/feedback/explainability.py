def explain(metric, weighted_score):
    if metric == "fluency":
        if weighted_score >= 30:
            return "Speech flow and pace are good."
        elif weighted_score >= 20:
            return "Some pauses or fillers affected fluency."
        else:
            return "Fluency needs improvement."

    if metric == "grammar":
        if weighted_score >= 25:
            return "Grammar usage is mostly accurate."
        elif weighted_score >= 15:
            return "Some grammatical errors detected."
        else:
            return "Frequent grammatical mistakes detected."

    if metric == "confidence":
        if weighted_score >= 30:
            return "You sound confident while speaking."
        elif weighted_score >= 20:
            return "Confidence is moderate."
        else:
            return "Hesitation reduced confidence."

    if metric == "topic":
        return "Response is reasonably on topic."

    if metric == "interview":
        return "Interview-style response evaluated."

    return "No explanation available."

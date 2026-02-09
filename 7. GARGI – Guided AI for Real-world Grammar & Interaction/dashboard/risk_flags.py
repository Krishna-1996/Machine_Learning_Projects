def compute_risk_flags(scores, trends):
    flags = []

    if scores[-1] < 55:
        flags.append("Low overall performance")

    if any(v < 0 for v in trends.values()):
        flags.append("Recent decline detected")

    if len(scores) >= 3 and scores[-1] < scores[-2] < scores[-3]:
        flags.append("Consistent downward trend")

    return flags

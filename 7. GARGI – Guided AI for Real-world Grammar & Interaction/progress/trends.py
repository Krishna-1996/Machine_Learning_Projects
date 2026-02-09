def compute_trends(sessions):
    if len(sessions) < 2:
        return {}

    latest = sessions[-1]["components"]
    previous = sessions[-2]["components"]

    trends = {}
    for metric in latest:
        diff = round(latest[metric] - previous.get(metric, 0), 1)
        trends[metric] = diff

    return trends

from firebase.firestore_client import load_all_users
from statistics import mean
from progress.bands import score_band
from progress.trends import compute_trends

def learner_overview(user_id, sessions):
    scores = [s["total_score"] for s in sessions]
    latest = scores[-1]

    return {
        "user_id": user_id,
        "attempts": len(scores),
        "latest_score": latest,
        "band": score_band(latest),
        "avg_score": round(mean(scores), 1)
    }

def learner_detail(user_id, sessions):
    scores = [s["total_score"] for s in sessions]

    components_history = {}
    for session in sessions:
        for k, v in session["components"].items():
            components_history.setdefault(k, []).append(v)

    trends = compute_trends(sessions)
    return scores, components_history, trends

def class_overview():
    users = load_all_users()
    all_scores = []
    band_dist = {"A": 0, "B": 0, "C": 0, "D": 0}

    for sessions in users.values():
        latest = sessions[-1]["total_score"]
        all_scores.append(latest)
        band_dist[score_band(latest)] += 1

    return {
        "total_learners": len(users),
        "average_score": round(mean(all_scores), 1),
        "band_distribution": band_dist
    }

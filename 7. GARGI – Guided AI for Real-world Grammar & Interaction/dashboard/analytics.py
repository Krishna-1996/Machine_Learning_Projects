from pathlib import Path
import json
from statistics import mean

from progress.trends import compute_trends
from progress.bands import score_band

DATA_DIR = Path("progress_data")


def load_all_users():
    users = {}
    for file in DATA_DIR.glob("*.json"):
        user_id = file.stem
        users[user_id] = json.loads(file.read_text())
    return users


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


def class_overview(users):
    all_scores = []
    band_dist = {"A": 0, "B": 0, "C": 0, "D": 0}

    for sessions in users.values():
        latest = sessions[-1]["total_score"]
        all_scores.append(latest)
        band_dist[score_band(latest)] += 1

    return {
        "total_learners": len(users),
        "average_score": round(mean(all_scores), 1) if all_scores else 0,
        "band_distribution": band_dist
    }

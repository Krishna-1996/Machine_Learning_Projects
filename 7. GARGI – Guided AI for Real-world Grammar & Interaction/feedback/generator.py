from .templates import TEMPLATES

def generate_feedback(level: str, scores: dict) -> dict:
    return {
        metric: TEMPLATES[level][metric]
        for metric in scores.keys()
    }

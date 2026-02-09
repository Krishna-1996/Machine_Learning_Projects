TIPS = {
    "fluency": "Reduce filler words and maintain steady pace.",
    "grammar": "Focus on sentence structure and verb agreement.",
    "confidence": "Pause less and speak with steady volume.",
    "topic": "Stay focused on the question topic.",
    "interview": "Use situation–action–result structure."
}

def get_tip(metric):
    return TIPS.get(metric, "")

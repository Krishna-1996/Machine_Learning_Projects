from fastapi import FastAPI
from app.schemas import ScoreRequest, ScoreResponse
from app.scoring import fluency, grammar, relevance
from app.scoring.scoring_engine import calculate_total_score
from app.feedback.generator import generate_feedback

app = FastAPI(title="GARGI Scoring API", version="1.0")

@app.post("/score_text", response_model=ScoreResponse)
def score_text(req: ScoreRequest):
    f_score, f_exp = fluency.fluency_score(req.transcript, req.duration_seconds)
    g_score, g_exp = grammar.grammar_score(req.transcript)
    r_score, r_exp = relevance.relevance_score(req.transcript)

    scores = {
        "fluency": f_score,
        "grammar": g_score,
        "relevance": r_score
    }

    explainability = {
        "fluency": f_exp,
        "grammar": g_exp,
        "relevance": r_exp
    }

    total = calculate_total_score(scores, req.level)
    feedback = generate_feedback(req.level, scores)

    return ScoreResponse(
        level=req.level,
        scores=scores,
        total_score=total,
        explainability=explainability,
        feedback_tips=feedback
    )

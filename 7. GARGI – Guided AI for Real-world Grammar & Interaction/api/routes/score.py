from fastapi import APIRouter, UploadFile, File, Form
from speech_analysis.stage3_analysis import run_stage3
from scoring.scoring_engine import compute_final_score
from scoring.explainability import explainability_output
import shutil

router = APIRouter()

@router.post("/score/speech")
async def score_speech(
    level: str = Form(...),
    file: UploadFile = File(...)
):
    with open("speech.wav", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    analysis = run_stage3()
    transcript = analysis["grammar"].get("transcript", "")

    score, breakdown = compute_final_score(level, transcript, analysis)
    explain = explainability_output(level, score, breakdown)

    return {
        "level": level,
        "score": score,
        "breakdown": breakdown,
        "explainability": explain,
        "raw_metrics": analysis
    }

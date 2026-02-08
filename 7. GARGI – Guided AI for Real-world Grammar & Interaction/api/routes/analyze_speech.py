from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import shutil
from services.whisper_service import transcribe_audio
from speech_analysis.stage3_analysis import analyze_speech

router = APIRouter()

TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

@router.post("/analyze/speech")
async def analyze_speech_api(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a")):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    temp_path = os.path.join(TEMP_DIR, file.filename)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        transcript = transcribe_audio(temp_path)
        analysis = analyze_speech(temp_path, transcript)

        return {
            "transcript": transcript,
            **analysis
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

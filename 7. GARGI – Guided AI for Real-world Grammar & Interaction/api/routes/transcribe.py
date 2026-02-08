from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
from services.whisper_service import transcribe_audio

router = APIRouter()

UPLOAD_DIR = "temp_audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a")):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    temp_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        text = transcribe_audio(temp_path)
        return {"transcript": text}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

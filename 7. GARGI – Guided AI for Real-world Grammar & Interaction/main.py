from fastapi import FastAPI
from api.routes.evaluate import router as evaluate_router
from api.routes.transcribe import router as transcribe_router
from api.routes.analyze_speech import router as analyze_speech_router

app = FastAPI(
    title="GARGI Backend API",
    version="0.3.0"
)

app.include_router(evaluate_router)
app.include_router(transcribe_router)
app.include_router(analyze_speech_router)

@app.get("/health")
def health_check():
    return {"status": "ok"}

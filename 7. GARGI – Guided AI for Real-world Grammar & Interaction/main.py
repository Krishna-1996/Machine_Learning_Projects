from fastapi import FastAPI
from api.routes.score_speech import router as score_router
from api.routes.health import router as health_router

app = FastAPI(title="GARGI Speech Scoring API")

app.include_router(health_router)
app.include_router(score_router)

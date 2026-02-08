from fastapi import FastAPI
from api.routes.evaluate import router as evaluate_router

app = FastAPI(
    title="GARGI Backend API",
    version="0.1.0"
)

# Register routes
app.include_router(evaluate_router)

@app.get("/health")
def health_check():
    return {"status": "ok"}

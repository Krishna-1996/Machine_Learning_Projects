FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements-cloud.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source
WORKDIR /a
COPY . /app

# Cloud Run listens on $PORT
ENV PORT=8080
EXPOSE 8080

# IMPORTANT: match your FastAPI app object path
# Based on your code: api/app.py defines `app = FastAPI(...)`
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]
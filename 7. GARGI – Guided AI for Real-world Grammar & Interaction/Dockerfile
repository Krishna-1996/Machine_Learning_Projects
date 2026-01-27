FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements-cloud.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Cloud Run provides PORT env var; default is 8080
CMD exec uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8080}
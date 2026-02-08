# GARGI API Dockerfile (Cloud Run compatible)

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
  && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements-cloud.txt /app/requirements.txt

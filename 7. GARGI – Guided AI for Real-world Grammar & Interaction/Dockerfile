# GARGI API Dockerfile (Cloud Run compatible)
FROM python:3.13-slim

# Environment variables 
ENV PYTHONUNBUFFERED=1
# # Cloud Run default port
ENV PORT=8080 
# Prevents Python from writing .pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1  

# Working directory
WORKDIR /app

# Install system dependencies (curl, if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app files
COPY . /app

# Start the FastAPI app using Uvicorn
CMD exec uvicorn api.app:app --host 0.0.0.0 --port ${PORT}

FROM python:3.10-slim

WORKDIR /app

# Minimal system deps
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first
COPY requirements-cloud.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

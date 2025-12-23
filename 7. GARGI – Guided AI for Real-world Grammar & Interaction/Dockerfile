FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (lightweight). libgomp helps some ML wheels; curl for health/debug.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app

EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# GARGI — Guided AI for Real-world General Interaction

🎯 **GARGI (User Perspective)** is a **local-first**, **explainable** AI coach that evaluates spoken or written responses and provides actionable feedback for **real-world communication** (not exam-only speaking).  
It focuses on **fluency**, **grammar**, **filler usage**, **topic alignment**, and **guided improvement over time**.

🎯 **GARGI (Developer Perspective)** is a modular, end-to-end AI system that:
1. Generates realistic speaking topics  
2. Captures speech and converts it to text  
3. Analyzes fluency, grammar, and hesitation  
4. Evaluates semantic topic relevance  
5. Produces explainable scores and coaching  
6. Logs learning sessions over time  
7. Visualizes progress via a dashboard  
8. Exposes functionality through a REST API  
9. Runs fully containerized using Docker  

---

## 🚀 Core Capabilities

- 🎤 Speech & Text Evaluation  
- 🧠 Semantic Topic Relevance (Embeddings-based)  
- ✍️ Grammar Analysis (LanguageTool)  
- 📊 Learning Progress Dashboard  
- 🌐 REST API (FastAPI)  
- 🐳 Dockerized Deployment  
- 🧩 Explainable AI Feedback (XAI)

---

## Key Features

- **Offline-first pipeline**: Runs locally on Windows (privacy-first and low-cost)
- **Explainability (XAI)**: Transparent scoring trace (base score + penalties) with evidence (WPM, pause ratio, grammar rules, semantic similarity)
- **Topic enrichment**: Separates prompt wording from topic meaning using metadata-aware topic datasets
- **Semantic topic relevance**: Similarity, coverage, sentence-level on-topic ratio, and anchor-based rubric
- **Coaching layer**: Prioritized feedback, concrete improvement actions, reflection prompts, and confidence estimation
- **Progress tracking**: Streamlit dashboard powered by append-only session logs
- **API layer (FastAPI)**: Product-ready interface for future web, mobile, or cloud deployment

---

## System Overview

User
|
+-- Speech / Text Input
|
+-- Stage 1: Speech Capture (CLI)
+-- Stage 2: Transcription (Whisper)
+-- Stage 3: Fluency + Grammar Analysis
+-- Stage 4: Scoring & Explainability
+-- Stage 5: Topic Relevance (Embeddings)
+-- Stage 6: Coaching & Confidence
+-- Stage 7: Learning Dashboard
+-- Stage 8: FastAPI + Docker
`-- Stage 9: (Optional) Cloud, Auth, Multi-user

---

## Detailed Pipeline Explanation

### Stage 0 — Topic Dataset Enrichment (Offline Preprocessing)
**Purpose:** Separate *what the topic is about* from *how it is phrased*.

- Input: `topics.csv`
- Output: enriched topic objects containing:
  - `instruction`
  - `topic_content`
  - `topic_type`
  - `constraints`
  - `expected_anchors`
  - `topic_keyphrases`

**Why this matters:**  
Enables semantic evaluation and prevents penalizing paraphrasing or creative phrasing.

---

### Stage 1 — Speech Input & Transcription
**Purpose:** Capture natural spoken responses.

- Local audio recording (fixed duration)
- Whisper-based transcription
- English language gate

**Why this matters:**  
Ensures evaluation reflects real-world speaking conditions rather than typed answers.

---

### Stage 2 — Topic Selection (Metadata-aware)
**Purpose:** Provide meaningful and structured speaking prompts.

- Random or category-based topic selection
- Returns:
  - Human-readable topic text
  - Structured `topic_obj` for evaluation

---

### Stage 3 — Speech Analysis
**Purpose:** Quantify speaking mechanics.

- **Fluency**
  - Words per minute (WPM)
  - Pause ratio
- **Fillers**
  - Hesitation markers (e.g., “uh”, “um”, repetitions)
- **Grammar**
  - LanguageTool (local server)
  - Schema-stable fallback mode if unavailable

---

### Stage 4 — Scoring & Explainability
**Purpose:** Convert raw metrics into transparent, explainable scores.

- Scores (0–10):
  - Fluency
  - Grammar
  - Fillers
  - Overall
- Outputs:
  - Scoring trace (base score + penalties)
  - Evidence used for each deduction

**Why this matters:**  
Users can audit *why* they received a particular score.

---

### Stage 5 — Topic Relevance (Semantic Evaluation)
**Purpose:** Measure meaning alignment, not keyword overlap.

- Embedding similarity between topic meaning and response
- Semantic coverage of expected topic keyphrases
- Sentence-level on-topic ratio with dynamic thresholds
- Anchor-based rubric bonus for expected structure

---

### Stage 6 — Learning Guidance & Trust
**Purpose:** Transform evaluation into coaching.

- Confidence score with explanation
- Top improvement priorities
- Concrete next actions
- Reflection prompts
- Append-only session logging (`sessions/sessions.jsonl`)

---

### Stage 7 — Learning Progress Dashboard
**Purpose:** Visualize learning and improvement over time.

- Streamlit-based dashboard
- Reads from session history
- Displays score trends, fluency improvements, and consistency

---

### Stage 8 — API + Docker

#### Stage 8.1 — FastAPI Layer
- `GET /topics` — returns topic object and prompt text
- `POST /evaluate/text` — evaluates text input and optionally logs the session

#### Stage 8.2 — Docker Deployment
- Fully containerized application
- Reproducible local deployment
- API verified via Swagger UI

---

## Project Structure (Suggested)

GARGI/
├── api/ # FastAPI application
├── coaching/ # Stage 6 coaching logic
├── core/ # Shared configuration
├── dashboard/ # Stage 7 Streamlit dashboard
├── scoring_feedback/ # Stage 4 scoring logic
├── services/ # External services (LanguageTool)
├── sessions/ # Persistent learning history
├── speech_analysis/ # Stage 3 analysis
├── speech_input/ # Stage 1 audio capture
├── topic_generation/ # Topic creation & enrichment
├── topic_relevance/ # Stage 5 semantic evaluation
├── tools/ # Topic enrichment utilities
├── Dockerfile
├── docker-compose.yml
├── main.py # CLI pipeline
├── README.md
└── requirements.txt


---

## Setup (Windows 11)

### 1) Create a virtual environment
Python 3.10 is recommended.
```bash
    python -m venv .venv
    .venv\Scripts\activate
---
### 2) Install dependencies
```bash

pip install -r requirements.txt
```
### 3) (Optional) Start LanguageTool server
```bash

java -jar languagetool-server.jar --port 8081
```
If LanguageTool is not running, GARGI continues in fallback grammar mode and reports this in the evidence.

Run the CLI Pipeline
```bash
python main.py
```
Records audio
Transcribes speech
Evaluates Stages 3–6
Appends a session entry to sessions/sessions.jsonl

### 4) Run the Dashboard (Stage 7)
```bash
streamlit run dashboard/stage7_dashboard.py
```

### 5) Run the API (Stage 8.1)
```bash
uvicorn api.app:app --reload --port 8000
Swagger UI: http://127.0.0.1:8000/docs

OpenAPI spec: http://127.0.0.1:8000/openapi.json

Recommended workflow:

GET /topics

POST /evaluate/text
```

### 6) Run with Docker (Stage 8.2)
```bash
docker compose up --build
```

## Roadmap
**Stage 9 (Optional):** Cloud deployment (GCP / Cloud Run), authentication, multi-user support
Deferred by design — GARGI is fully functional without paid cloud services

**Stage 10:** CI/CD with GitHub Actions (tests, linting, Docker build)

**Stage 11:** Android app (local-first mode, optional cloud sync later)

**Stage 12+:** Personalization, agentic coaching, long-term learner modeling, and infrastructure-as-code (Terraform)

## Notes on Trust & Correctness
GARGI emphasizes transparent evidence, auditable scoring traces, and explainable metrics (WPM, pause ratio, grammar rules, semantic similarity and coverage).
Future improvements include benchmarking against human ratings and automated regression tests to ensure scoring stability.

## License
This project is licensed under the MIT License.

You are free to use, modify, and distribute this software for personal or commercial purposes, provided that the original copyright
and license notice are included.

See the <LICENSE> file for full details.
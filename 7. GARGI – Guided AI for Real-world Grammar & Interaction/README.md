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
## 🤖Android App
→ FastAPI Backend
→ AI Evaluation Pipeline
→ Scoring + Explainability
→ Coaching & Logging
→ Dashboard / Cloud

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

## System Architecture (Pipeline View)

```text
GARGI System
├── AI / ML Pipeline (Stage 1 - 6 Completed)
│   ├── Speech transcription (Whisper)
│   ├── Fluency analysis (WPM, pauses)
│   ├── Grammar analysis (LanguageTool)
│   ├── Semantic relevance (embeddings)
│   ├── Explainable scoring
│   └── Coaching & confidence estimation
│ 
├── Backend (FastAPI) (Stage 7-8 Completed)
│   ├── Tech Stack
│   │   ├── Python
│   │   ├── FastAPI
│   │   └── Docker
│   ├── Endpoints
│   │   ├── `GET /topics`
│   │   └── `POST /evaluate/text`
│   ├── Security
│   │   ├── API Key
│   │   └── Optional Basic Auth
│ 
├── Android Application (Stage 9 Completed)
│   ├── Tech Stack
│   │   ├── Kotlin
│   │   ├── Jetpack Compose
│   │   ├── MVVM
│   │   ├── Hilt
│   │   └── Retrofit / OkHttp
│   ├── Capabilities
│   │   ├── On-device speech recording
│   │   ├── Topic fetch via API
│   │   ├── Text evaluation via API
│   │   ├── Score visualization
│   │   ├── Feedback rendering
│   │   └── API key authentication
│
├── Google Cloud Platform Integration (Stage 10 – Upcoming)
│   ├── Services Used
│   │   ├── Vertex AI
│   │   ├── Cloud Run
│   │   ├── Cloud Storage
│   │   ├── Secret Manager
│   │   └── IAM
│   ├── Goals
│   │   ├── Scalable AI inference
│   │   ├── Secure API access
│   │   ├── Model lifecycle management
│   │   └── Android-cloud connectivity
│
├── CI/CD (Stage 12 – Planned)
│   ├── GitHub Actions
│   ├── Automated tests
│   ├── Docker builds
│   └── Cloud Run deployment
│
├── Advanced Android (Stage 13 – Planned)
│   ├── Multi-user authentication
│   ├── Offline caching
│   ├── Cloud sync
│   ├── Advanced UI/UX
│   └── Personalized learning views
│
└── Future Extensions
    ├── Stage 14+: Personalization, AgenticAI coaching, IaC (Infrastructure as Code) via Terraform
    └── Stage 15+: More AI integrations, UX enhancements

```
---
## Why This Project Matters

GARGI demonstrates:
- Full-stack AI engineering
- Explainable ML design
- Mobile + backend integration
- Cloud-native thinking
- Real-world product engineering

This project is suitable for **AI/ML Engineer**, **Applied Scientist**, and **Data Engineer** roles.

---

## Setup (Windows 11)

### 1) Create a virtual environment
Python 3.10 is recommended.
```bash
    python -m venv .venv
    .venv\Scripts\activate
```
### 2) Install dependencies
```bash

pip install -r requirements.txt
```
### 3) (Optional) Start LanguageTool server
```bash

java -jar languagetool-server.jar --port 8081
```
If LanguageTool is not running, **GARGI** continues in fallback grammar mode and reports this in the evidence.

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


## Notes on Trust & Correctness
**GARGI** emphasizes transparent evidence, auditable scoring traces, and explainable metrics (WPM, pause ratio, grammar rules, semantic similarity and coverage).
Future improvements include benchmarking against human ratings and automated regression tests to ensure scoring stability.

## License
This project is licensed under the MIT License.

You are free to use, modify, and distribute this software for personal or commercial purposes, provided that the original copyright
and license notice are included.

See the **LICENSE** file for full details.
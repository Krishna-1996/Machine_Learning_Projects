# GARGI — Guided AI for Real-world General Interaction

GARGI is a **local-first**, **explainable** AI coach that evaluates spoken responses and provides actionable feedback for real-world communication (not exam-only speaking).  
It focuses on: **fluency**, **grammar**, **filler usage**, **topic alignment**, and **guided improvement over time**.

**🚀 Core Capabilities**

- 🎤 Speech & Text Evaluation
- 🧠 Semantic Topic Relevance (Embeddings-based)
- ✍️ Grammar Analysis (LanguageTool)
- 📊 Learning Progress Dashboard
- 🌐 REST API (FastAPI)
- 🐳 Dockerized Deployment
- 🧩 Explainable AI Feedback

## Key Features
- **Offline-first pipeline**: runs locally on Windows (privacy + low cost)
- **Explainability (XAI)**: scoring trace (base + penalties) and evidence (WPM, pause ratio, grammar rules, similarity)
- **Topic enrichment**: separates prompt wording from topic meaning using a metadata-aware topic dataset
- **Semantic topic relevance**: similarity, coverage, sentence-level on-topic ratio, and anchor rubric
- **Coaching layer**: priorities + actions + reflection prompts + confidence estimation
- **Progress tracking**: Streamlit dashboard powered by append-only session logs
- **API layer (FastAPI)**: product-ready interface for future cloud/mobile deployment

**User**
 ├─ Speech / Text Input
 │
 ├─ Stage 1: Speech Capture (CLI)
 ├─ Stage 2: Transcription (Whisper)
 ├─ Stage 3: Fluency + Grammar Analysis
 ├─ Stage 4: Scoring & Explainability
 ├─ Stage 5: Topic Relevance (Embeddings)
 ├─ Stage 6: Coaching & Confidence
 ├─ Stage 7: Learning Dashboard
 ├─ Stage 8: FastAPI + Docker
 └─ Stage 9: (Planned) Auth, Users, Cloud


## Pipeline Stages (Current)
**Stage 0 — Topic Dataset Enrichment (offline preprocessing)**
- Input: `topics.csv`
- Output: enriched topic objects with:
  - `instruction`, `topic_content`, `topic_type`, `constraints`
  - `expected_anchors`, `topic_keyphrases`

**Stage 1 — Speech Input & Transcription**
- Record audio (fixed duration) → `speech.wav`
- Whisper transcription → `transcription.txt`
- Language gate (English)

**Stage 2 — Topic Selection (metadata-aware)**
- Random topic selection with optional category filter
- Returns a structured `topic_obj`

**Stage 3 — Speech Analysis**
- Fluency: WPM, pause ratio, filler counts
- Grammar: LanguageTool (local server) with schema-stable fallback mode

**Stage 4 — Scoring & Explainability**
- Scores (0–10): Fluency, Grammar, Fillers, Overall
- Outputs scoring trace + evidence used

**Stage 5 — Topic Relevance**
- Embedding similarity (topic meaning vs response)
- Semantic coverage (topic phrases vs response keyphrases)
- Sentence-level on-topic ratio + dynamic threshold
- Anchor rubric bonus (expected structural elements)

**Stage 6 — Learning Guidance & Trust**
- Confidence score + explanation
- Top 3 priorities with concrete actions
- Reflection prompts
- Append-only session logging: `sessions/sessions.jsonl`

**Stage 7 — Learning Progress Dashboard**
- Streamlit dashboard: trends and session table from `sessions/sessions.jsonl`

**Stage 8.1 — FastAPI Layer**
- `GET /topics` → returns `topic_obj` + `topic_text`
- `POST /evaluate/text` → runs Stages 3–6 on text input and optionally appends session history

## Project Structure (suggested)
```
GARGI/
  ├── api/                  # FastAPI application
  ├── coaching/             # Stage 6 coaching logic
  ├── core/                 # Paths & shared config
  ├── dashboard/            # Stage 7 Streamlit dashboard
  ├── scoring_feedback/     # Stage 4 scoring
  ├── services/             # External services (LanguageTool)
  ├── sessions/             # Persistent learning history
  ├── speech_analysis/      # Stage 3 analysis
  ├── speech_input/         # Stage 1 audio capture
  ├── topic_generation/     # Topic creation & enrichment
  ├── topic_relevance/      # Stage 5 semantic evaluation
  ├── tools/                # Topic enrichment utilities
  ├── Dockerfile
  ├── docker-compose.yml
  ├── main.py               # CLI pipeline
  ├── README.md
  ├── requirements.txt

```

## Setup (Windows 11)
### 1) Create a virtual environment
Python 3.10 is recommended for maximal compatibility (3.13 can be used if your audio/whisper stack supports it).
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) (Optional) Start LanguageTool server for grammar
If you use local LanguageTool:
```bash
java -jar languagetool-server.jar --port 8081
```
If LanguageTool is not running, GARGI continues with **fallback grammar mode** and includes a warning in evidence.

## Run the CLI (main.py)
```bash
python main.py
```
This records audio, transcribes it, evaluates it (Stages 3–6), and appends a session row to:
- `sessions/sessions.jsonl`

## Run the Dashboard (Stage 7)
```bash
streamlit run dashboard/stage7_dashboard.py
```
The dashboard reads `sessions/sessions.jsonl` and visualizes progress over time.

## Run the API (Stage 8.1)
```bash
uvicorn api.app:app --reload --port 8000
```
Open:
- Swagger: `http://127.0.0.1:8000/docs`
- OpenAPI: `http://127.0.0.1:8000/openapi.json`

Recommended API workflow:
1. `GET /topics`
2. `POST /evaluate/text` using the returned `topic_obj` and your transcript

## Roadmap
- **Stage 8.2**: Docker (portable deployment)
- **Stage 9**: Cloud deployment (Google Cloud / Cloud Run), model storage (GCS)
- **Stage 10**: CI/CD (GitHub Actions: tests + build + deploy)
- **Stage 11**: Android app (multi-user accounts, profiles, and session sync)
- **Stage 12+**: Personalization, agentic coaching, and infrastructure-as-code (Terraform)

## Notes on Trust & Correctness
GARGI uses **transparent evidence** (WPM, pause ratio, grammar rules, semantic similarity/coverage) and a **scoring trace** to make outputs auditable.  
Future improvements include benchmarking against human ratings and automated regression tests to ensure scoring stability.

## License
Add a license of your choice (MIT is common for open-source prototypes).

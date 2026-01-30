# GARGI — Guided AI for Real-world General Interaction

**GARGI** is an AI-powered speaking coach designed for **real-world communication — not exams**.

It helps users practice speaking on realistic topics and receive **honest, explainable feedback** on fluency, grammar, fillers, pacing, and topic relevance.  
GARGI works across **Android, Cloud, and API-based platforms**, with a live production backend on **Google Cloud Run + Vertex AI**.

---

## 🎯 What Problem GARGI Solves

Most speaking apps:
- Focus on time spoken, not *what was said*
- Provide generic or black-box scores
- Optimize for exams, not real conversations

**GARGI is different.**

It evaluates **meaning, clarity, relevance, and delivery**, and explains *why* a score was given — just like a human coach would.

---

## 🌍 What GARGI Does (User Perspective)

Users can:

- 🎤 Practice speaking on **real-world topics**
- 🧠 Get AI feedback on **what they said**, not just duration
- 📊 Understand fluency, grammar, fillers, pauses, and topic relevance
- 🔁 Improve over time with **structured, actionable guidance**
- 📱 Use it anywhere — **only an internet connection is required**

---

## 🛠️ What GARGI Is (Engineering Perspective)

GARGI is a **production-grade, full-stack AI system** that:

1. Generates realistic speaking topics  
2. Captures speech on Android devices  
3. Converts speech → text (on device)  
4. Analyzes fluency, grammar, fillers, and pauses  
5. Measures semantic topic relevance using embeddings  
6. Produces **explainable scores and coaching feedback**  
7. Exposes functionality via a secure REST API  
8. Runs globally on **Google Cloud Run**  
9. Uses **Vertex AI** for scalable AI inference  

This is not a prototype — it is a **live, deployed system**.

---

## 📱 Android App (Live)

**Architecture Flow**
`
Android App (Jetpack Compose)
        ↓ 
      HTTPS
        ↓
FastAPI Backend (Cloud Run)
        ↓
Vertex AI (Embeddings + LLMs)
`

### Android Tech Stack
- Kotlin
- Jetpack Compose
- MVVM Architecture
- Hilt (Dependency Injection)
- Retrofit + OkHttp
- On-device speech recognition

### Current Capabilities
- Topic selection via API
- Speech capture
- Transcript generation
- AI evaluation via Cloud Run
- Score and feedback visualization
- Global usability (UK, India, anywhere)

---

## ☁️ Cloud & AI Stack (Live)

### Backend
- FastAPI
- Google Cloud Run
- API-key protected endpoints
- OpenAPI / Swagger enabled

### AI
- Vertex AI Embeddings  
  (semantic relevance, topic alignment)
- Explainable scoring logic  
  (fluency, grammar, fillers, pacing)

**Privacy-first design**
- Audio is processed on device
- Only text is sent to the cloud
- No audio is stored remotely

---

## 🚀 Core Capabilities

- 🎤 Speech evaluation (Android)
- 🧠 Semantic topic relevance (Vertex AI embeddings)
- ✍️ Grammar & fluency metrics
- 📊 Explainable scoring (not black-box)
- 🌐 REST API (FastAPI)
- ☁️ Cloud-native deployment (Cloud Run)
- 🔐 API security
- 🌍 Global availability

---

## 🧩 Explainability First (XAI)

GARGI does **not** just return a score.

Every evaluation includes:
- Speaking rate (WPM)
- Pause patterns
- Grammar signals
- Topic similarity ratios
- Sentence-level relevance
- Concrete improvement advice

This makes feedback:
- **Auditable**
- **Trustworthy**
- **Actionable**

---

## 🧠 System Architecture (High-Level)
`
GARGI Platform
├── Android Client
│ ├── Speech capture
│ ├── Transcript buffer
│ ├── Evaluation UI
│ └── History (planned)
│
├── Backend API (FastAPI)
│ ├── Topic service
│ ├── Evaluation orchestrator
│ ├── Session logging
│ └── Security layer
│
├── Vertex AI
│ ├── Text embeddings (live)
│ └── Gemini LLMs (planned)
│
└── Google Cloud
├── Cloud Run
├── Logging & monitoring
├── Billing & quotas
└── IAM / Secrets
`

---

## 🧪 Live API (Public)

**Swagger UI**  
https://gargi-api-59813842911.asia-south1.run.app/docs

**Endpoints**
- `GET /health`
- `GET /topics`
- `POST /evaluate/text`

---

## 🧭 Roadmap (What’s Coming Next)

### 🔹 Phase 1 — Android Speech UX
- Continuous listening (no 60-second limit)
- Pause / resume speaking
- Auto-pause after inactivity
- Transcript continuity across pauses

### 🔹 Phase 2 — Smarter AI Evaluation
- Integrate Vertex AI Gemini
- Topic-aware reasoning
- Honest, non-generic feedback
- Improved fluency & grammar critique
- Controlled cost within free credits

### 🔹 Phase 3 — User System
- Email + password authentication
- Phone number login
- Multi-user support on one device
- Secure identity management

### 🔹 Phase 4 — History & Dashboard
- Speaking history
- Topic history
- Score trends over time
- Personalized improvement insights

### 🔹 Phase 5 — UI / UX Polish
- Prime-grade design
- Improved visual feedback
- Accessibility improvements
- Web application version

---

## 💡 Why This Project Matters

GARGI demonstrates:

- Real-world AI product thinking
- Cloud-native backend design
- Android + backend integration
- Explainable AI (XAI)
- Responsible AI usage
- Cost-aware AI deployment
- End-to-end system ownership

This project reflects skills relevant to:
- **AI / ML Engineer**
- **Applied Scientist**
- **Mobile + Backend Engineer**
- **Cloud Engineer**

---

## 🛡️ Cost & Safety

- Uses Vertex AI embeddings (low cost)
- Cloud Run free-tier friendly
- Billing alerts enabled
- No audio stored in the cloud
- Secure API access

---

## 📄 License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this software for personal or commercial purposes, provided that the original copyright
and license notice are included.

See the **LICENSE** file for full details.

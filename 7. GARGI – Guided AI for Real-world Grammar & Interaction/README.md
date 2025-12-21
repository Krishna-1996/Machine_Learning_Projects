# GARGI — Guided AI for Real-world General Interaction

GARGI is an explainable, offline-first AI system designed to evaluate spoken responses and provide actionable feedback for **real-world spoken communication**.

The project focuses on **general interaction** rather than exam-specific scoring, helping users improve clarity, fluency, grammar, and topic alignment in everyday conversations, academic discussions, interviews, and professional settings.

---

## 🔍 Key Capabilities

- 🎙️ **Speech Input & Transcription**
  - Local audio recording
  - Speech-to-text using Whisper

- 🗣️ **Fluency Analysis**
  - Speaking rate (WPM)
  - Pause ratio
  - Filler word detection

- ✍️ **Grammar Analysis**
  - Rule-based grammar checking via LanguageTool
  - Error density and explainable grammar feedback

- 🎯 **Topic Relevance & Alignment**
  - Semantic similarity using Sentence Transformers
  - Concept-level coverage analysis
  - Sentence-level on-topic ratio
  - Explainable relevance diagnostics

- 🧠 **Explainability & Trust Layer**
  - Transparent scoring logic
  - Evidence-based feedback
  - XAI-inspired scoring traces

- 📈 **Learning-Oriented Feedback**
  - Priority improvement suggestions
  - Coaching-style guidance
  - Reflection prompts for self-assessment

---

## 🧩 System Architecture


---

## 🛠️ Technology Stack

- Python 3.10 / 3.13
- Whisper (speech-to-text)
- LanguageTool (grammar analysis)
- Sentence Transformers (`all-mpnet-base-v2`)
- YAKE (keyword extraction)
- NumPy, SciPy, scikit-learn

All components are **free**, **local-first**, and compatible with **Windows**.

---

## 🚀 Running the Project

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Start LanguageTool server:
    java -jar languagetool-server.jar --port 8081

3. Run GARGI:
    python main.py

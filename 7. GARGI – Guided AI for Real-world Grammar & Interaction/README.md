# GARGI — Guided AI for Real-World General Interaction

GARGI is an end-to-end, offline-first AI system designed to evaluate spoken responses and provide explainable, learning-oriented feedback on **fluency, grammar, fillers, and topic relevance**.

The project is built with a strong emphasis on **interpretability**, **trust**, and **human-centric feedback**, making it suitable for speaking practice, interview preparation, and language assessment research.

---

## 🔍 Key Features

- 🎙️ **Speech Input & Transcription**
  - Records user speech locally
  - Transcribes using OpenAI Whisper (offline-capable)

- 🗣️ **Fluency Analysis**
  - Speaking rate (WPM)
  - Pause ratio
  - Filler word detection

- ✍️ **Grammar Analysis**
  - Rule-based grammar checking (LanguageTool)
  - Error density and explainable error categories

- 🎯 **Topic Relevance Evaluation**
  - Semantic similarity using Sentence Transformers
  - Concept coverage analysis
  - Sentence-level relevance ratio
  - Explainable relevance feedback

- 🧠 **Explainability & Trust (XAI-inspired)**
  - Scoring traces
  - Evidence-based feedback
  - Clear reasoning for every score

- 📈 **Learning-Oriented Feedback**
  - Actionable suggestions
  - Priority improvement areas
  - Reflection prompts for self-assessment

---

## 🧩 System Architecture


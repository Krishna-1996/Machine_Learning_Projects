import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from utils import load_sessions

st.set_page_config(
    page_title="GARGI Learning Dashboard",
    layout="wide"
)

st.title("📈 GARGI — Learning Progress Dashboard")
st.caption("Guided AI for Real-world General Interaction")

df = load_sessions()

if df.empty:
    st.warning("No session data found. Complete some GARGI sessions first.")
    st.stop()

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("Filters")

date_range = st.sidebar.date_input(
    "Select date range",
    value=(df["timestamp"].min().date(), df["timestamp"].max().date())
)

if len(date_range) == 2:
    start, end = date_range
    df = df[
        (df["timestamp"].dt.date >= start) &
        (df["timestamp"].dt.date <= end)
    ]

# -------------------------------
# High-level Metrics
# -------------------------------
st.subheader("Overall Progress Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Sessions", len(df))
col2.metric("Avg Overall Score", round(df["scores.overall"].mean(), 2))
col3.metric("Avg Relevance", round(df["relevance.relevance_score"].mean(), 2))
col4.metric("Avg Confidence", round(df["confidence.confidence_score"].mean(), 2))

# -------------------------------
# Trend Charts
# -------------------------------
st.subheader("Score Trends Over Time")

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(df["timestamp"], df["scores.overall"], label="Overall")
ax.plot(df["timestamp"], df["scores.fluency"], label="Fluency")
ax.plot(df["timestamp"], df["scores.grammar"], label="Grammar")
ax.plot(df["timestamp"], df["scores.fillers"], label="Fillers")

ax.set_xlabel("Session Time")
ax.set_ylabel("Score")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# -------------------------------
# Relevance & Confidence Trends
# -------------------------------
st.subheader("Topic Alignment & Confidence")

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(df["timestamp"], df["relevance.relevance_score"], label="Relevance")
ax2.plot(df["timestamp"], df["confidence.confidence_score"], label="Confidence")

ax2.set_xlabel("Session Time")
ax2.set_ylabel("Score")
ax2.legend()
ax2.grid(True)

st.pyplot(fig2)

# -------------------------------
# Error Trends
# -------------------------------
st.subheader("Error Reduction Over Time")

fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(df["timestamp"], df["grammar.error_count"], label="Grammar Errors")
ax3.plot(df["timestamp"], df["fillers.total"], label="Filler Words")

ax3.set_xlabel("Session Time")
ax3.set_ylabel("Count")
ax3.legend()
ax3.grid(True)

st.pyplot(fig3)

# -------------------------------
# Session Table
# -------------------------------
st.subheader("Session History")

columns = [
    "timestamp",
    "topic_raw",
    "scores.overall",
    "relevance.relevance_score",
    "confidence.confidence_score"
]

st.dataframe(df[columns].sort_values("timestamp", ascending=False))

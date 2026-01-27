import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import math

from utils import load_sessions

st.set_page_config(page_title="GARGI Learning Dashboard", layout="wide")

st.title("GARGI — Learning Progress Dashboard")
st.caption("Guided AI for Real-world General Interaction")

df = load_sessions()

if df.empty:
    st.warning("No session data found. Run GARGI sessions first to generate sessions/sessions.jsonl.")
    st.stop()

# -------------------------------
# Helpers (convert numpy -> python types)
# -------------------------------
def to_py(x, ndigits: int | None = 2):
    """Convert pandas/numpy scalars to native Python types, handle NaN."""
    if x is None:
        return None
    try:
        # pandas may store as numpy scalars
        if pd.isna(x):
            return None
    except Exception:
        pass

    # Convert numpy scalars to Python
    if hasattr(x, "item"):
        try:
            x = x.item()
        except Exception:
            pass

    # Round floats if requested
    if isinstance(x, float) and ndigits is not None:
        if math.isnan(x):
            return None
        return round(x, ndigits)

    return x


def delta(curr, prev, ndigits: int = 2):
    if prev is None:
        return None
    c = to_py(curr, ndigits=None)
    p = to_py(prev, ndigits=None)
    if c is None or p is None:
        return None
    try:
        return to_py(c - p, ndigits=ndigits)
    except Exception:
        return None


# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("Filters")

min_d = df["timestamp"].min().date()
max_d = df["timestamp"].max().date()

date_range = st.sidebar.date_input("Select date range", value=(min_d, max_d))
window = st.sidebar.slider("Rolling window (sessions)", min_value=2, max_value=10, value=3)

if len(date_range) == 2:
    start, end = date_range
    df = df[(df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)]

if len(df) == 0:
    st.warning("No sessions in the selected date range.")
    st.stop()

# -------------------------------
# Latest Session Summary + Deltas
# -------------------------------
st.subheader("Latest Session Snapshot")

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) >= 2 else None

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Latest Overall",
    to_py(latest.get("scores.overall")),
    delta(to_py(latest.get("scores.overall"), ndigits=None), None if prev is None else prev.get("scores.overall"))
)
c2.metric(
    "Latest Relevance",
    to_py(latest.get("relevance.relevance_score")),
    delta(to_py(latest.get("relevance.relevance_score"), ndigits=None), None if prev is None else prev.get("relevance.relevance_score"))
)
c3.metric(
    "Latest Grammar",
    to_py(latest.get("scores.grammar")),
    delta(to_py(latest.get("scores.grammar"), ndigits=None), None if prev is None else prev.get("scores.grammar"))
)
c4.metric(
    "Latest Fillers",
    to_py(latest.get("scores.fillers")),
    delta(to_py(latest.get("scores.fillers"), ndigits=None), None if prev is None else prev.get("scores.fillers"))
)
c5.metric(
    "Confidence",
    to_py(latest.get("confidence.confidence_score")),
    delta(to_py(latest.get("confidence.confidence_score"), ndigits=None), None if prev is None else prev.get("confidence.confidence_score"))
)

st.write(f"**Latest Topic:** {latest.get('topic_raw', '')}")
st.write(
    f"**Relevance Label:** {latest.get('relevance.label', 'N/A')} | "
    f"**Confidence Label:** {latest.get('confidence.label', 'N/A')}"
)

# -------------------------------
# Overall Summary
# -------------------------------
st.subheader("Progress Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sessions", int(len(df)))
col2.metric("Avg Overall", to_py(df["scores.overall"].mean()))
col3.metric("Avg Relevance", to_py(df["relevance.relevance_score"].mean()))
col4.metric("Avg Confidence", to_py(df["confidence.confidence_score"].mean()))

# -------------------------------
# Weakest area recommendation
# -------------------------------
st.subheader("Weakest Area (Focus Recommendation)")

areas = {
    "Fluency": df["scores.fluency"].mean(),
    "Grammar": df["scores.grammar"].mean(),
    "Fillers": df["scores.fillers"].mean(),
    "Topic Relevance": df["relevance.relevance_score"].mean() * 10.0,  # scale to 0–10
}
weak_area = min(areas, key=lambda k: (areas[k] if not pd.isna(areas[k]) else 999))
st.info(f"Recommended focus: **{weak_area}** (lowest average in selected range).")

# -------------------------------
# Rolling averages
# -------------------------------
st.subheader("Rolling Trend (Smoothed)")

roll = df.copy()
roll["overall_roll"] = roll["scores.overall"].rolling(window).mean()
roll["relevance_roll"] = roll["relevance.relevance_score"].rolling(window).mean()
roll["grammar_roll"] = roll["scores.grammar"].rolling(window).mean()
roll["fillers_roll"] = roll["scores.fillers"].rolling(window).mean()
roll["fluency_roll"] = roll["scores.fluency"].rolling(window).mean()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(roll["timestamp"], roll["overall_roll"], label="Overall (roll)")
ax.plot(roll["timestamp"], roll["fluency_roll"], label="Fluency (roll)")
ax.plot(roll["timestamp"], roll["grammar_roll"], label="Grammar (roll)")
ax.plot(roll["timestamp"], roll["fillers_roll"], label="Fillers (roll)")
ax.set_xlabel("Time")
ax.set_ylabel("Score")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# -------------------------------
# Raw score trends
# -------------------------------
st.subheader("Raw Score Trends")

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(df["timestamp"], df["scores.overall"], label="Overall")
ax2.plot(df["timestamp"], df["scores.fluency"], label="Fluency")
ax2.plot(df["timestamp"], df["scores.grammar"], label="Grammar")
ax2.plot(df["timestamp"], df["scores.fillers"], label="Fillers")
ax2.set_xlabel("Time")
ax2.set_ylabel("Score")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# -------------------------------
# Relevance & Confidence
# -------------------------------
st.subheader("Topic Alignment & Confidence")

fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(df["timestamp"], df["relevance.relevance_score"], label="Relevance")
ax3.plot(df["timestamp"], df["confidence.confidence_score"], label="Confidence")
ax3.set_xlabel("Time")
ax3.set_ylabel("Score")
ax3.legend()
ax3.grid(True)
st.pyplot(fig3)

# -------------------------------
# Evidence trends
# -------------------------------
st.subheader("Evidence Trends (What drives the scores)")

e1, e2 = st.columns(2)

with e1:
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(df["timestamp"], df["evidence.wpm"], label="WPM")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Words per minute")
    ax4.grid(True)
    st.pyplot(fig4)

with e2:
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    ax5.plot(df["timestamp"], df["evidence.pause_ratio"], label="Pause Ratio")
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Pause ratio")
    ax5.grid(True)
    st.pyplot(fig5)

e3, e4 = st.columns(2)

with e3:
    fig6, ax6 = plt.subplots(figsize=(10, 4))
    ax6.plot(df["timestamp"], df["grammar.error_count"], label="Grammar Errors")
    ax6.set_xlabel("Time")
    ax6.set_ylabel("Count")
    ax6.grid(True)
    st.pyplot(fig6)

with e4:
    fig7, ax7 = plt.subplots(figsize=(10, 4))
    ax7.plot(df["timestamp"], df["fillers.total"], label="Filler Total")
    ax7.set_xlabel("Time")
    ax7.set_ylabel("Count")
    ax7.grid(True)
    st.pyplot(fig7)

# -------------------------------
# Session Table
# -------------------------------
st.subheader("Session History")

cols = [
    "timestamp",
    "topic_raw",
    "scores.overall",
    "scores.fluency",
    "scores.grammar",
    "scores.fillers",
    "relevance.relevance_score",
    "relevance.label",
    "confidence.confidence_score",
    "evidence.wpm",
    "evidence.pause_ratio",
    "grammar.error_count",
    "fillers.total",
]
existing = [c for c in cols if c in df.columns]
st.dataframe(df[existing].sort_values("timestamp", ascending=False), use_container_width=True)

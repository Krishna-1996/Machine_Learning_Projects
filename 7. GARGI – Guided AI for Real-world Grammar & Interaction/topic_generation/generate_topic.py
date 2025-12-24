"""
Topic Generation Utilities
Project: GARGI — Guided AI for Real-world General Interaction

Provides:
- load_topics(): reads topics_enriched.csv (cached)
- list_categories(): returns available categories
- get_random_topic(category): returns a normalized topic_obj
- search_topics(query, category, limit): fast typeahead suggestions

Expected CSV columns (recommended):
topic_id, category, topic_raw, instruction, topic_content, topic_type,
constraints, expected_anchors, topic_keyphrases
"""

from __future__ import annotations

import os
from typing import Optional, List, Dict, Any
import pandas as pd

TOPICS_FILE = "topics_enriched.csv"

# In-memory cache to avoid reloading CSV on every call
_CACHE_DF: Optional[pd.DataFrame] = None


def load_topics(force_reload: bool = False) -> pd.DataFrame:
    """
    Load topics from CSV and cache them for fast repeated access.
    """
    global _CACHE_DF

    if _CACHE_DF is not None and not force_reload:
        return _CACHE_DF

    if not os.path.exists(TOPICS_FILE):
        raise FileNotFoundError(
            f"Missing {TOPICS_FILE}. Run: python tools/enrich_topics.py"
        )

    df = pd.read_csv(TOPICS_FILE, encoding="utf-8")

    # Ensure expected columns exist (defensive)
    required_cols = [
        "topic_id", "category", "topic_raw", "instruction",
        "topic_content", "topic_type", "constraints",
        "expected_anchors", "topic_keyphrases"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    # Normalize types for safe filtering/search
    df["category"] = df["category"].fillna("").astype(str)
    df["topic_raw"] = df["topic_raw"].fillna("").astype(str)
    df["topic_content"] = df["topic_content"].fillna("").astype(str)
    df["topic_type"] = df["topic_type"].fillna("general").astype(str)
    df["instruction"] = df["instruction"].fillna("").astype(str)

    _CACHE_DF = df
    return df


def _to_list(val) -> List[str]:
    """
    Convert pipe-separated strings like "a|b|c" into ["a","b","c"].
    Handles NaN/None safely.
    """
    if val is None:
        return []
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return []
    return [x.strip() for x in s.split("|") if x.strip()]


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a row dict to the stable topic_obj format used throughout GARGI.
    """
    row = dict(row)

    # Convert pipe-separated columns into lists
    row["constraints"] = _to_list(row.get("constraints", ""))
    row["expected_anchors"] = _to_list(row.get("expected_anchors", ""))
    row["topic_keyphrases"] = _to_list(row.get("topic_keyphrases", ""))

    # Guarantee minimum required fields
    row.setdefault("topic_type", "general")
    if not row.get("topic_content"):
        row["topic_content"] = row.get("topic_raw", "") or ""

    # Standardize missing category/text
    row["category"] = (row.get("category") or "").strip()
    row["topic_raw"] = (row.get("topic_raw") or "").strip()
    row["topic_content"] = (row.get("topic_content") or "").strip()

    return row


# --------------------------
# Public API (used by CLI/API/UI)
# --------------------------
def list_categories() -> List[str]:
    """
    Return all available topic categories.
    """
    df = load_topics()
    cats = sorted(
        [c for c in df["category"].dropna().unique().tolist() if str(c).strip()]
    )
    return cats


def get_categories() -> List[str]:
    """
    Backward-compatible alias (your existing code calls this).
    """
    return list_categories()


def get_random_topic(category: Optional[str] = None) -> Dict[str, Any]:
    """
    Return a random topic row (optionally filtered by category),
    normalized into a consistent topic_obj schema.
    """
    df = load_topics()

    if category and category.strip():
        cat = category.strip().lower()
        df2 = df[df["category"].str.lower() == cat]
        if df2.empty:
            df2 = df
    else:
        df2 = df

    row = df2.sample(1).iloc[0].to_dict()
    return _normalize_row(row)


def search_topics(
    query: str,
    category: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Fast topic search for typeahead suggestions.
    - query: user typed text (recommend: trigger when len>=3)
    - category: optional category filter
    - limit: maximum number of suggestions
    Returns a list of normalized topic_obj dicts.
    """
    q = (query or "").strip().lower()
    if len(q) < 2:
        return []

    df = load_topics()

    # Optional category filtering
    if category and category.strip():
        cat = category.strip().lower()
        df = df[df["category"].str.lower() == cat]
        if df.empty:
            return []

    # Search across topic_raw + topic_content
    haystack = (df["topic_raw"] + " " + df["topic_content"]).str.lower()
    mask = haystack.str.contains(q, na=False)

    hits = df[mask].head(int(limit))
    return [_normalize_row(r.to_dict()) for _, r in hits.iterrows()]


def get_topic_by_id(topic_id: int) -> Optional[Dict[str, Any]]:
    """
    Optional helper: fetch a topic by topic_id (useful for API/app deep links).
    """
    df = load_topics()
    if "topic_id" not in df.columns:
        return None

    try:
        tid = int(topic_id)
    except Exception:
        return None

    rows = df[df["topic_id"].astype(str) == str(tid)]
    if rows.empty:
        return None

    return _normalize_row(rows.iloc[0].to_dict())

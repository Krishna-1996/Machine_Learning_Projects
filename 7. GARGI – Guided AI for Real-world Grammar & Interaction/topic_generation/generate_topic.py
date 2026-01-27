from __future__ import annotations

import csv
import os
import random
from typing import Any, Dict, List, Optional

# If your CSV path is different, update here.
TOPICS_FILE = os.getenv("TOPICS_FILE", "topics_enriched.csv")

# In-memory cache (loaded once per container instance)
_CACHE: Optional[List[Dict[str, Any]]] = None


def _load_topics() -> List[Dict[str, Any]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    if not os.path.exists(TOPICS_FILE):
        raise FileNotFoundError(
            f"Missing {TOPICS_FILE}. Ensure it is copied into the container image "
            f"or set TOPICS_FILE env var to its path."
        )

    rows: List[Dict[str, Any]] = []
    with open(TOPICS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Normalize keys/values
            row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items() if k}
            if row:
                rows.append(row)

    if not rows:
        raise RuntimeError(f"{TOPICS_FILE} loaded but contains 0 rows.")

    _CACHE = rows
    return rows


def get_categories() -> List[str]:
    """
    Returns unique categories from the topics file.
    Expects a 'category' column in topics_enriched.csv.
    """
    rows = _load_topics()
    cats = sorted({(r.get("category") or "").strip() for r in rows if (r.get("category") or "").strip()})
    return cats


def get_random_topic(category: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns a random topic row (dict) from topics_enriched.csv.
    If category is provided, samples only within that category.

    Expected columns (flexible):
      - category
      - topic_raw or topic
      - topic_content (optional)
      - topic_type (optional)
      - expected_anchors (optional)
      - topic_keyphrases (optional)

    Your API can post-process this dict as needed.
    """
    rows = _load_topics()

    if category:
        category = category.strip()
        filtered = [r for r in rows if (r.get("category") or "").strip() == category]
        if not filtered:
            raise ValueError(f"No topics found for category='{category}'")
        row = random.choice(filtered)
    else:
        row = random.choice(rows)

    # Build a stable schema for downstream code
    topic_raw = row.get("topic_raw") or row.get("topic") or row.get("topic_text") or ""
    topic_content = row.get("topic_content") or topic_raw
    topic_type = row.get("topic_type") or "general"

    # Some CSVs store lists as strings; keep as raw and let API parse if needed.
    expected_anchors = row.get("expected_anchors") or []
    topic_keyphrases = row.get("topic_keyphrases") or []

    return {
        "category": row.get("category") or "",
        "topic_raw": topic_raw,
        "topic_content": topic_content,
        "topic_type": topic_type,
        "expected_anchors": expected_anchors,
        "topic_keyphrases": topic_keyphrases,
        # Keep original row for debugging / future use
        "_row": row,
    }
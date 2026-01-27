from __future__ import annotations

import csv
import os
import random
from typing import Any, Dict, List, Optional

TOPICS_FILE = os.getenv("TOPICS_FILE", "topics_enriched.csv")
_CACHE: Optional[List[Dict[str, Any]]] = None


def _split_pipe(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    s = str(value).strip()
    if not s:
        return []
    return [p.strip() for p in s.split("|") if p.strip()]


def _load_topics() -> List[Dict[str, Any]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    if not os.path.exists(TOPICS_FILE):
        raise FileNotFoundError(
            f"Missing {TOPICS_FILE}. Ensure it is included in the container build context "
            f"(Docker COPY) or set TOPICS_FILE env var to its path."
        )

    rows: List[Dict[str, Any]] = []
    with open(TOPICS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items() if k}
            if row:
                rows.append(row)

    if not rows:
        raise RuntimeError(f"{TOPICS_FILE} loaded but contains 0 rows.")

    _CACHE = rows
    return rows


def get_categories() -> List[str]:
    rows = _load_topics()
    cats = sorted({(r.get("category") or "").strip() for r in rows if (r.get("category") or "").strip()})
    return cats


def get_random_topic(category: Optional[str] = None) -> Dict[str, Any]:
    rows = _load_topics()

    if category:
        category = category.strip()
        filtered = [r for r in rows if (r.get("category") or "").strip() == category]
        if not filtered:
            raise ValueError(f"No topics found for category='{category}'")
        row = random.choice(filtered)
    else:
        row = random.choice(rows)

    topic_raw = row.get("topic_raw") or row.get("topic") or ""
    topic_content = row.get("topic_content") or topic_raw
    topic_type = row.get("topic_type") or "general"

    expected_anchors = _split_pipe(row.get("expected_anchors"))
    topic_keyphrases = _split_pipe(row.get("topic_keyphrases"))

    return {
        "topic_id": row.get("topic_id") or "",
        "category": (row.get("category") or "").strip(),
        "topic_raw": topic_raw,
        "instruction": row.get("instruction") or "",
        "topic_content": topic_content,
        "topic_type": topic_type,
        "constraints": row.get("constraints") or "",
        "expected_anchors": expected_anchors,
        "topic_keyphrases": topic_keyphrases,
    }
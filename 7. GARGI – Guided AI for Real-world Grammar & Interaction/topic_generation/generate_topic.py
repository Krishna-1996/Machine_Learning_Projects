from __future__ import annotations

import csv
import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

# This should be the enriched file produced by tools/enrich_topics.py
TOPICS_FILE = "topics_enriched.csv"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _topics_path() -> Path:
    # Prefer explicit env var if you ever want to mount topics elsewhere later
    p = (os.getenv("TOPICS_FILE_PATH", "") or "").strip()
    if p:
        return Path(p)
    return _project_root() / TOPICS_FILE


def _split_pipe(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split("|") if x.strip()]


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Output schema used across API/Android:
    {
      topic_id, category, topic_raw, instruction, topic_content, topic_type,
      constraints, expected_anchors (list), topic_keyphrases (list)
    }
    """
    out = dict(row or {})
    out["topic_id"] = (out.get("topic_id") or "").strip()
    out["category"] = (out.get("category") or "").strip()
    out["topic_raw"] = (out.get("topic_raw") or "").strip()
    out["instruction"] = (out.get("instruction") or "").strip()
    out["topic_content"] = (out.get("topic_content") or out["topic_raw"]).strip()
    out["topic_type"] = (out.get("topic_type") or "general").strip() or "general"
    out["constraints"] = (out.get("constraints") or "").strip()

    out["expected_anchors"] = _split_pipe(out.get("expected_anchors", ""))
    out["topic_keyphrases"] = _split_pipe(out.get("topic_keyphrases", ""))

    return out


@lru_cache(maxsize=1)
def _load_topics() -> List[Dict[str, Any]]:
    path = _topics_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Ensure topics_enriched.csv exists in project root "
            f"or set TOPICS_FILE_PATH env var."
        )

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(_normalize_row(r))
    if not rows:
        raise RuntimeError(f"{path} is empty or could not be parsed.")
    return rows


def get_categories() -> List[str]:
    rows = _load_topics()
    cats = sorted({r.get("category", "").strip() for r in rows if r.get("category", "").strip()})
    return cats


def get_random_topic(category: Optional[str] = None) -> Dict[str, Any]:
    rows = _load_topics()
    if category and category.strip():
        cat = category.strip().lower()
        pool = [r for r in rows if (r.get("category", "").lower() == cat)]
        if not pool:
            pool = rows
    else:
        pool = rows
    return random.choice(pool)


def search_topics(query: str, category: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
    q = (query or "").strip().lower()
    if len(q) < 2:
        return []

    rows = _load_topics()

    if category and category.strip():
        cat = category.strip().lower()
        rows = [r for r in rows if r.get("category", "").lower() == cat]

    hits = []
    for r in rows:
        hay = f"{r.get('topic_raw','')} {r.get('topic_content','')}".lower()
        if q in hay:
            hits.append(r)
            if len(hits) >= int(limit):
                break
    return hits


def get_topic_by_id(topic_id: int) -> Optional[Dict[str, Any]]:
    tid = str(topic_id).strip()
    if not tid:
        return None
    for r in _load_topics():
        if str(r.get("topic_id", "")).strip() == tid:
            return r
    return None
import os
from typing import Optional, List, Dict, Any
import pandas as pd

TOPICS_FILE = "topics_enriched.csv"

# Small cache so we don't reload CSV on every request
_CACHE_DF: Optional[pd.DataFrame] = None

def load_topics() -> pd.DataFrame:
    if not os.path.exists(TOPICS_FILE):
        raise FileNotFoundError(
            f"Missing {TOPICS_FILE}. Run: python tools/enrich_topics.py"
        )
    return pd.read_csv(TOPICS_FILE, encoding="utf-8")

def get_categories() -> list[str]:
    df = load_topics()
    cats = sorted(df["category"].dropna().unique().tolist())
    return cats

def get_random_topic(category: str | None = None) -> dict:
    df = load_topics()

    if category:
        df2 = df[df["category"].str.lower() == category.lower()]
        if len(df2) == 0:
            df2 = df
    else:
        df2 = df

    row = df2.sample(1).iloc[0].to_dict()

    # Normalize pipe-separated fields to Python lists
    def to_list(val):
        if val is None:
            return []
        s = str(val).strip()
        if not s:
            return []
        return [x.strip() for x in s.split("|") if x.strip()]

    row["constraints"] = to_list(row.get("constraints", ""))
    row["expected_anchors"] = to_list(row.get("expected_anchors", ""))
    row["topic_keyphrases"] = to_list(row.get("topic_keyphrases", ""))

    return row

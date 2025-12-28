# D:\Machine_Learning_Projects\7. GARGI – Guided AI for Real-world Grammar & Interaction\api\routes\topics.py
from __future__ import annotations

from typing import Any, Dict, Optional, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# Ensure project root is on sys.path for absolute imports (same pattern as evaluate.py)
import api.deps  # noqa: F401


router = APIRouter(tags=["topics"])


# -----------------------------
# Response models (matches your current backend JSON used by Android)
# -----------------------------
class TopicResponse(BaseModel):
    topic_obj: Dict[str, Any] = Field(default_factory=dict)
    topic_text: str = Field(default="")


class CategoriesResponse(BaseModel):
    categories: List[str] = Field(default_factory=list)


# -----------------------------
# Internal helpers
# -----------------------------
def _topic_text_from_obj(topic_obj: Dict[str, Any]) -> str:
    """
    Your topic_generation output may contain any of these keys depending on version.
    We normalize to a user-facing string.
    """
    for k in ("topic_raw", "topic_text", "topic_content", "prompt", "topic"):
        v = topic_obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _get_topic_provider():
    """
    Prefer the real pipeline from topic_generation/generate_topic.py.
    Falls back safely with clear error messages.
    """
    try:
        from topic_generation.generate_topic import get_random_topic, get_categories  # type: ignore
        return get_random_topic, get_categories
    except Exception as e:
        raise ImportError(
            "Could not import topic providers from topic_generation.generate_topic. "
            "Expected functions: get_random_topic(category=None), get_categories(). "
            f"Original error: {e}"
        )


# -----------------------------
# Routes
# -----------------------------
@router.get("/categories", response_model=CategoriesResponse)
def categories() -> CategoriesResponse:
    """
    Returns categories from topics_enriched.csv (via topic_generation.generate_topic.get_categories()).
    """
    try:
        _, get_categories = _get_topic_provider()
        cats = get_categories()
        cats = [c for c in cats if isinstance(c, str) and c.strip()]
        return CategoriesResponse(categories=sorted(set(cats)))
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load categories: {e}")


@router.get("/topics", response_model=TopicResponse)
def topics(
    category: Optional[str] = Query(default=None, description="Optional topic category"),
) -> TopicResponse:
    """
    Returns ONE random topic, optionally filtered by category.

    This replaces the dummy 'Example topic' response.
    """
    try:
        get_random_topic, _ = _get_topic_provider()

        topic_obj = get_random_topic(category=category)
        if not isinstance(topic_obj, dict):
            raise HTTPException(status_code=500, detail="Topic provider returned non-dict topic_obj")

        topic_text = _topic_text_from_obj(topic_obj)
        if not topic_text:
            # Even if the object exists, ensure topic_text is never empty for Android UI.
            topic_text = "Topic found, but text was missing in topic_obj."

        return TopicResponse(topic_obj=topic_obj, topic_text=topic_text)

    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate topic: {e}")

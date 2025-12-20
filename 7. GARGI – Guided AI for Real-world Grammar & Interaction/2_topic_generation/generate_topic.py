"""
Stage 2: Topic Generation
Project: GARGI
Author: Krishna
"""

import csv
import random
import logging
from typing import Optional, Dict

TOPIC_FILE = "topics.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_topics(filepath: str):
    topics = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            topics.append(row)
    if not topics:
        raise ValueError("Topic file is empty.")
    return topics

def get_random_topic(
    category: Optional[str] = None,
    seed: Optional[int] = None
) -> Dict:
    topics = load_topics(TOPIC_FILE)

    if category:
        topics = [t for t in topics if t["category"] == category]
        if not topics:
            raise ValueError(f"No topics found for category: {category}")

    if seed is not None:
        random.seed(seed)

    selected = random.choice(topics)
    return selected

def main():
    try:
        topic = get_random_topic()
        logging.info("Selected Topic:")
        logging.info(f"Topic: {topic['topic']}")
        logging.info(f"Category: {topic['category']}")

        # Save topic for later stages
        with open("selected_topic.txt", "w", encoding="utf-8") as f:
            f.write(topic["topic"])

    except Exception as e:
        logging.error(f"Stage 2 failed: {e}")

if __name__ == "__main__":
    main()

import yake

kw_extractor = yake.KeywordExtractor(top=5, stopwords=None)

def relevance_score(transcript: str):
    keywords = kw_extractor.extract_keywords(transcript)

    score = min(1.0, len(keywords) / 5)

    explanation = (
        f"Identified keywords: "
        + ", ".join(k for k, _ in keywords)
        if keywords else
        "Low topical signal detected."
    )

    return round(score, 2), explanation

from scoring.scoring_engine import compute_final_score

# Mock analysis output (Sprint-3 style)
mock_analysis = {
    "fluency": {
        "wpm": 120,
        "pause_ratio": 0.12,
        "filler_words": {
            "um": 2,
            "uh": 1
        }
    },
    "grammar": {
        "total_errors": 3
    }
}

def test_beginner_score():
    total, components = compute_final_score(
        level="beginner",
        transcript="I like playing football because it is fun and healthy.",
        analysis=mock_analysis
    )

    print("TOTAL:", total)
    print("COMPONENTS:", components)

    assert isinstance(total, int)
    assert total > 0
    assert "fluency" in components
    assert "confidence" in components
    assert "topic" in components
    
if __name__ == "__main__":
    test_beginner_score()

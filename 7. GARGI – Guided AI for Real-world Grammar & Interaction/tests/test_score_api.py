import requests

BASE_URL = "http://127.0.0.1:8080"

def test_health():
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200
    print("Health endpoint OK:", resp.json())

def test_score_empty_analysis():
    payload = {
        "user_id": "test_user",
        "level": "beginner",
        "transcript": "hello world",
        "analysis": {}
    }
    resp = requests.post(f"{BASE_URL}/score", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    print("Score response:", data)

def test_score_partial_analysis():
    payload = {
        "user_id": "test_user2",
        "level": "beginner",
        "transcript": "hello world this is a test",
        "analysis": {"fluency": 0.8, "grammar": 0.5}
    }
    resp = requests.post(f"{BASE_URL}/score", json=payload)
    assert resp.status_code == 200
    print("Partial analysis OK:", resp.json())

if __name__ == "__main__":
    test_health()
    test_score_empty_analysis()
    test_score_partial_analysis()

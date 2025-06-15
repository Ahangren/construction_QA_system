import requests
import json

BASE_URL = "http://localhost:8000"

def test_search():
    payload = {
        "query": "混凝土强度",
        "top_k": 3,
        "metadata_filter": {"source": "GB"}
    }
    response = requests.post(f"{BASE_URL}/search", json=payload)
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print(response.json())

if __name__ == "__main__":
    test_health()
    test_search()
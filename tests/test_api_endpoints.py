import requests
import pytest

BASE_URL = "http://localhost:8000"

def test_health():
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "sentiment-analysis"

def test_status():
    resp = requests.get(f"{BASE_URL}/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "audio_processor" in data
    assert "transcription" in data
    assert "sentiment_analysis" in data
    assert "storage" in data

def test_text_sentiment():
    payload = {"text": "I love this product!"}
    resp = requests.post(f"{BASE_URL}/analyze/text", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "sentiment" in data
    assert data["sentiment"]["overall_sentiment"] in ["positive", "neutral", "negative"] 
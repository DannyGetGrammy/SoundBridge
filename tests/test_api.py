"""FastAPI tests for the SoundBridge API skeleton."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["project"] == "SoundBridge"
    assert "artifacts" in payload


def test_tracks() -> None:
    response = client.get("/tracks?limit=5")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) <= 5
    assert payload
    assert {"track_id", "genre", "processed_path"}.issubset(payload[0])


def test_genres() -> None:
    response = client.get("/genres")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, dict)
    assert payload


def test_audio_search_baseline() -> None:
    response = client.post(
        "/search/audio",
        json={"track_id": "1482", "method": "baseline", "top_k": 5},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["method"] == "baseline"
    assert len(payload["results"]) == 5
    assert payload["results"][0]["track_id"] != "1482"


def test_metrics() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    payload = response.json()
    assert payload["baseline_mean_precision_at_k"] is not None
    assert payload["clap_mean_precision_at_k"] is not None

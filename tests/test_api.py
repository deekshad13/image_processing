import pytest
import io
from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)


# ── Health Check ──────────────────────────────────────────────────────────────
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ── Gallery ───────────────────────────────────────────────────────────────────
def test_get_gallery():
    response = client.get("/gallery")
    assert response.status_code == 200
    data = response.json()
    assert "symptoms" in data
    assert len(data["symptoms"]) == 10


def test_get_symptom_detail():
    response = client.get("/gallery/Galls")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Galls"
    assert "description" in data
    assert "severity" in data
    assert "action" in data


def test_get_symptom_detail_not_found():
    response = client.get("/gallery/FakeSymptom")
    assert response.status_code == 404


# ── Compare — Invalid Inputs ──────────────────────────────────────────────────
def test_compare_invalid_format():
    fake_pdf = io.BytesIO(b"%PDF-1.4 fake pdf content")
    response = client.post(
        "/compare",
        files={"file": ("test.pdf", fake_pdf, "application/pdf")}
    )
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]


def test_compare_corrupted_image():
    corrupted = io.BytesIO(b"this is not an image")
    response = client.post(
        "/compare",
        files={"file": ("corrupt.jpg", corrupted, "image/jpeg")}
    )
    assert response.status_code == 400


# ── Compare — Valid Image ─────────────────────────────────────────────────────
def test_compare_valid_image():
    with open("data/raw/Galls/" + __import__('os').listdir("data/raw/Galls")[0], "rb") as f:
        response = client.post(
            "/compare",
            files={"file": ("test.jpg", f, "image/jpeg")}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["matches"]) > 0
    assert "thresholds" in data
    assert data["matches"][0]["symptom"] is not None
    assert 0 <= data["matches"][0]["similarity_pct"] <= 100

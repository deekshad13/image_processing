import pytest
import io
import os
from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)


# ── Health Check ─────────────────────────────────────────────────────────────

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model"] in ("fine-tuned", "baseline")
    assert data["symptoms_loaded"] > 0


# ── Gallery ──────────────────────────────────────────────────────────────────

def test_get_gallery():
    response = client.get("/gallery")
    assert response.status_code == 200
    data = response.json()
    assert "symptoms" in data
    assert "total_images" in data
    assert len(data["symptoms"]) == 10
    for symptom in data["symptoms"]:
        assert "symptom_id" in symptom
        assert "display_name" in symptom
        assert "image_count" in symptom


def test_get_symptom_detail_valid():
    response = client.get("/gallery/galls")
    assert response.status_code == 200
    data = response.json()
    assert data["symptom_id"] == "galls"
    assert data["display_name"] == "Galls"
    assert data["image_count"] > 0


def test_get_symptom_detail_not_found():
    response = client.get("/gallery/fake_symptom")
    assert response.status_code == 404


# ── Verify — Invalid Inputs ─────────────────────────────────────────────────

def test_verify_invalid_symptom_id():
    fake_img = io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    response = client.post(
        "/verify/nonexistent_symptom",
        files={"file": ("test.jpg", fake_img, "image/jpeg")},
    )
    assert response.status_code == 404


def test_verify_invalid_format():
    fake_pdf = io.BytesIO(b"%PDF-1.4 fake pdf content")
    response = client.post(
        "/verify/galls",
        files={"file": ("test.pdf", fake_pdf, "application/pdf")},
    )
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]


def test_verify_corrupted_image():
    corrupted = io.BytesIO(b"this is not an image")
    response = client.post(
        "/verify/galls",
        files={"file": ("corrupt.jpg", corrupted, "image/jpeg")},
    )
    assert response.status_code == 400


# ── Verify — Valid Image ────────────────────────────────────────────────────

def _get_test_image_path():
    galls_dir = "data/raw/Galls"
    images = [f for f in os.listdir(galls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return os.path.join(galls_dir, images[0])


def test_verify_matching_symptom():
    img_path = _get_test_image_path()
    with open(img_path, "rb") as f:
        response = client.post(
            "/verify/galls",
            files={"file": ("test.jpg", f, "image/jpeg")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    result = data["result"]
    assert result["symptom_id"] == "galls"
    assert result["symptom_name"] == "Galls"
    assert 0 <= result["similarity_pct"] <= 100
    assert result["confidence"] in ("high", "medium", "low")
    assert result["recommendation"] is not None
    assert result["gallery_size"] > 0
    assert result["top_n_used"] > 0
    assert "thresholds" in data


def test_verify_different_symptom():
    img_path = _get_test_image_path()
    with open(img_path, "rb") as f:
        response = client.post(
            "/verify/leaf_color_change",
            files={"file": ("test.jpg", f, "image/jpeg")},
        )
    assert response.status_code == 200
    data = response.json()
    result = data["result"]
    assert result["symptom_id"] == "leaf_color_change"
    assert 0 <= result["similarity_pct"] <= 100


# ── Deprecated Endpoint ──────────────────────────────────────────────────────

def test_deprecated_compare_returns_410():
    fake_img = io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    response = client.post(
        "/compare",
        files={"file": ("test.jpg", fake_img, "image/jpeg")},
    )
    assert response.status_code == 410


# ── Response Schema Validation ───────────────────────────────────────────────

def test_verify_response_schema():
    img_path = _get_test_image_path()
    with open(img_path, "rb") as f:
        response = client.post(
            "/verify/galls",
            files={"file": ("test.jpg", f, "image/jpeg")},
        )
    data = response.json()
    assert set(data.keys()) == {"status", "result", "thresholds"}
    result_keys = {"symptom_id", "symptom_name", "similarity_pct", "confidence",
                   "recommendation", "gallery_size", "top_n_used"}
    assert set(data["result"].keys()) == result_keys

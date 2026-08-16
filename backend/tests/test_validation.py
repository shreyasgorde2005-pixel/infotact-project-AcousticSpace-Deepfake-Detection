from fastapi.testclient import TestClient
from app.main import app
from app.core.history import log_prediction, get_history

client = TestClient(app)

def test_cors_headers_present():
    response = client.options(
        "/api/v1/predict",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code in [200, 204]
    assert "access-control-allow-origin" in response.headers

def test_reject_unsupported_extensions_detail():
    unsupported_exts = ["doc.pdf", "image.png", "data.csv", "archive.zip"]
    for fname in unsupported_exts:
        response = client.post(
            "/api/v1/predict",
            files={"file": (fname, b"dummy content", "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

def test_reject_oversize_payload():
    large_payload = b"0" * (21 * 1024 * 1024)
    response = client.post(
        "/api/v1/predict",
        files={"file": ("giant.wav", large_payload, "audio/wav")},
    )
    assert response.status_code == 400
    assert "File too large" in response.json()["detail"]

def test_history_logging_order():
    log_prediction("test_real.wav", False, 0.95)
    log_prediction("test_fake.wav", True, 0.88)
    history = get_history()
    assert len(history) >= 2
    assert history[0]["filename"] == "test_fake.wav"
    assert history[1]["filename"] == "test_real.wav"

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_rejects_bad_extension():
    response = client.post(
        "/api/v1/predict",
        files={"file": ("notes.txt", b"not audio", "text/plain")},
    )
    assert response.status_code == 400

def test_predict_rejects_corrupted_audio():
    response = client.post(
        "/api/v1/predict",
        files={"file": ("fake.wav", b"this is not a real wav file", "audio/wav")},
    )
    assert response.status_code == 422

def test_history_returns_list():
    response = client.get("/history")
    assert response.status_code == 200
    assert "history" in response.json()
    assert isinstance(response.json()["history"], list)

def test_predict_returns_breathing_score():
    with open("../ml/data/demo_test_set/sample1_pair14_matched_room_10.wav", "rb") as f:
        response = client.post("/api/v1/predict", files={"file": ("test.wav", f, "audio/wav")})
    assert response.status_code == 200
    data = response.json()
    assert data["breathing_score"] is not None

def test_predict_response_fields_not_null():
    file_path = "../ml/data/demo_test_set/sample1_pair14_matched_room_10.wav"
    with open(file_path, "rb") as f:
        response = client.post("/api/v1/predict", files={"file": ("test.wav", f, "audio/wav")})
    assert response.status_code == 200
    data = response.json()
    assert data["breathing_score"] is not None
    assert data["rir_mismatch_score"] is not None

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

sample_input = {
    "worker_exp_yr": 5,
    "rain_mm_week": 120,
    "slope_deg": 20,
    "temp_c": 25,
    "bush_age_yr": 15,
    "hours_worked": 8,
    "flush": "first",
    "section": "Lower Garden"
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_invalid_working_hours():
    data = sample_input.copy()
    data["hours_worked"] = 25

    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_negative_rainfall():
    data = sample_input.copy()
    data["rain_mm_week"] = -10

    response = client.post("/predict", json=data)
    assert response.status_code == 422

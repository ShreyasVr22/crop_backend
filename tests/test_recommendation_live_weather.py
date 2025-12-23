from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_recommend_with_coords_live_weather():
    """Integration test: post coordinates and soil data to trigger live weather fetch
    and ensure the recommendation endpoint returns a valid response.
    """
    payload = {
        "nitrogen": 50,
        "phosphorus": 20,
        "potassium": 40,
        "ph_level": 6.5,
        "soil_type": "Loamy",
        "soil_moisture": "Medium",
        "latitude": 12.97,
        "longitude": 77.59
    }

    resp = client.post("/api/crop/recommend", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert isinstance(data.get("best_crop"), str)
    assert isinstance(data.get("confidence_score"), (float, int))
    assert isinstance(data.get("top_5_crops"), list)

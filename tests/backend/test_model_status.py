from fastapi.testclient import TestClient


def test_health_separates_api_up_from_model_unavailable(client: TestClient) -> None:
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["model"]["available"] is False
    assert payload["model"]["status"] == "unavailable"

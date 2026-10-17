from fastapi.testclient import TestClient


def test_health_reports_ok_and_model_not_trained(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"]
    assert payload["version"]
    assert payload["model"]["available"] is False
    assert payload["model"]["status"] == "unavailable"
    assert payload["model"]["architecture"] == "fused"
    assert payload["model"]["model_version"] == ""
    assert payload["database"]["ok"] is True


def test_root_points_to_health(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["health"] == "/api/health"


def test_health_schema_keys(client: TestClient) -> None:
    payload = client.get("/api/health").json()
    assert set(payload.keys()) == {"status", "service", "version", "model", "produce_model", "database"}
    assert payload["produce_model"] == {
        "available": False,
        "status": "unavailable",
        "model_version": "",
        "confirmation_threshold": None,
    }
    assert set(payload["model"].keys()) == {
        "available",
        "status",
        "message",
        "architecture",
        "model_version",
    }

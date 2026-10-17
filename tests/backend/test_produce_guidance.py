"""Produce identity, combined labels, guidance catalogue and severity contract.

Real-bundle checks use one train sample (Fresh banana) and one validation sample
(Rotten apple). The test split is not touched.
"""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.services import produce as produce_service
from app.services.inference import classify_image
from app.services.produce import (
    CatalogueError,
    ProduceIdentity,
    build_guidance,
    combined_label,
    load_catalogue,
    parse_selected_produce,
    validate_catalogue,
)
ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data/Dataset/Fresh/FreshBanana/b_f856.png"
SPOILED_APPLE = ROOT / "data/Dataset/Rotten/RottenApple/a_r596.png"
SEVERITY_TEXT = "The current model classifies Fresh or Spoiled but does not measure the degree of spoilage."

FORBIDDEN_PHRASES = (
    "mold detected",
    "shelf life",
    "days until",
    "expire",
    "safe to eat",
    "% spoiled",
    "mild",
    "moderate",
    "severe",
    "taste it",
    "make it safe",
    "cooking",
    "peel",
)


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (24, 24), (20, 40, 10)).save(buffer, format="PNG")
    return buffer.getvalue()


def _all_points(guidance: dict[str, Any]) -> list[dict[str, Any]]:
    return guidance["general_advice"] + guidance["storage"] + guidance["checks"]


# ------------------------------------------------------------------ catalogue


def test_catalogue_is_valid_and_complete() -> None:
    catalogue = load_catalogue()
    assert catalogue["catalogue_version"]
    assert len(catalogue["produce"]) == 12
    for produce_id in catalogue["produce"]:
        for freshness in ("Fresh", "Spoiled"):
            guidance = build_guidance(produce_id, freshness)
            assert 3 <= len(_all_points(guidance)) <= 5, (produce_id, freshness)
            assert guidance["sources"], (produce_id, freshness)
    for freshness in ("Fresh", "Spoiled"):
        assert build_guidance("unknown", freshness)["sources"]


def test_every_source_is_documented() -> None:
    for source in load_catalogue()["sources"].values():
        assert source["url"].startswith("https://")
        assert source["scope"] and source["publisher"] and source["title"]
        assert source["reviewed_on"] == "2026-10-16"


def test_guidance_avoids_banned_claims() -> None:
    catalogue = load_catalogue()
    for produce_id in [*catalogue["produce"], "unknown"]:
        for freshness in ("Fresh", "Spoiled"):
            text = " ".join(point["text"] for point in _all_points(build_guidance(produce_id, freshness))).lower()
            for phrase in FORBIDDEN_PHRASES:
                assert phrase not in text, (produce_id, freshness, phrase)


def test_malformed_catalogue_is_rejected() -> None:
    base = json.loads(json.dumps(load_catalogue()))
    broken = copy.deepcopy(base)
    broken["produce"]["banana"]["storage"][0]["sources"] = ["not_a_source"]
    with pytest.raises(CatalogueError):
        validate_catalogue(broken)
    broken = copy.deepcopy(base)
    broken["produce"]["banana"]["storage"][0]["sources"] = []
    with pytest.raises(CatalogueError):
        validate_catalogue(broken)
    broken = copy.deepcopy(base)
    broken["produce"]["bitter_gourd"]["storage_review"] = "reviewed"
    with pytest.raises(CatalogueError):
        validate_catalogue(broken)
    broken = copy.deepcopy(base)
    broken["produce"]["unknown"] = broken["produce"]["banana"]
    with pytest.raises(CatalogueError):
        validate_catalogue(broken)


def test_guidance_matches_type_and_class() -> None:
    banana_fresh = build_guidance("banana", "Fresh")
    assert "room temperature" in banana_fresh["storage"][0]["text"]
    assert banana_fresh["review_status"] == "reviewed"

    apple_spoiled = build_guidance("apple", "Spoiled")
    assert "throw it away" in apple_spoiled["general_advice"][0]["text"]
    assert "Clean the spot" in apple_spoiled["storage"][0]["text"]
    assert not any("room temperature" in p["text"] for p in _all_points(apple_spoiled))

    strawberry = build_guidance("strawberry", "Fresh")
    assert "refrigerator" in strawberry["storage"][0]["text"]
    potato = build_guidance("potato", "Fresh")
    assert "light" in potato["storage"][0]["text"]

    tomato_spoiled = build_guidance("tomato", "Spoiled")
    assert any("soft produce" in p["text"] for p in tomato_spoiled["general_advice"])
    apple_mold = build_guidance("apple", "Spoiled")
    assert not any("soft produce" in p["text"] for p in apple_mold["general_advice"])


def test_unreviewed_produce_is_marked_not_invented() -> None:
    gourd = build_guidance("bitter_gourd", "Fresh")
    assert gourd["review_status"] == "awaiting_review"
    # Falls back to the generic reviewed refrigerator note, never a gourd-specific claim.
    assert "bitter" not in " ".join(p["text"] for p in gourd["storage"]).lower()

    unknown = build_guidance("unknown", "Fresh")
    assert unknown["review_status"] == "generic"


def test_spoiled_guidance_never_suggests_salvage() -> None:
    for produce_id in load_catalogue()["produce"]:
        text = " ".join(p["text"] for p in _all_points(build_guidance(produce_id, "Spoiled"))).lower()
        for phrase in ("cut off", "cut away", "trim", "wash"):
            assert phrase not in text, (produce_id, phrase)


# ------------------------------------------------------------------ identity


def test_parse_selected_produce() -> None:
    assert parse_selected_produce(None).source == "unknown"
    assert parse_selected_produce("").produce_id == "unknown"
    assert parse_selected_produce("unknown").source == "unknown"
    banana = parse_selected_produce("banana")
    assert banana.source == "user_selected" and banana.label == "Banana"
    assert parse_selected_produce(" banana ").produce_id == "banana"


@pytest.mark.parametrize(
    "bad",
    ["dragonfruit", "Banana", "ban ana", "banana;rm -rf", "<script>alert(1)</script>", "x" * 500, "../../etc/passwd"],
)
def test_unsupported_produce_is_rejected_without_echo(client: TestClient, bad: str) -> None:
    response = client.post(
        "/api/predict",
        data={"produce": bad},
        files={"file": ("s.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "unsupported_produce"
    assert bad not in response.text
    assert client.get("/api/guidance", params={"produce": bad}).status_code == 400


def test_combined_label_text() -> None:
    assert combined_label("Fresh", parse_selected_produce("banana")) == "Fresh banana"
    assert combined_label("Spoiled", parse_selected_produce("apple")) == "Spoiled apple"
    assert combined_label("Fresh", parse_selected_produce(None)) == "Fresh produce"
    assert combined_label("Spoiled", parse_selected_produce("unknown")) == "Spoiled produce"
    assert combined_label("Fresh", parse_selected_produce("capsicum")) == "Fresh bell pepper"


# ------------------------------------------------------------------ endpoints


def test_produce_endpoint_lists_options_and_user_selected_mode(client: TestClient) -> None:
    body = client.get("/api/produce").json()
    ids = [option["id"] for option in body["options"]]
    assert ids[-1] == "unknown"
    assert {"banana", "apple", "tomato", "bitter_gourd"} <= set(ids)
    assert body["identification"] == {
        "mode": "user_selected",
        "available": False,
        "model_version": None,
        "confirmation_threshold": None,
        "supported_ids": [],
    }
    assert body["catalogue_version"] == load_catalogue()["catalogue_version"]


def test_guidance_endpoint_works_without_a_model(client: TestClient) -> None:
    body = client.get("/api/guidance", params={"produce": "banana", "freshness": "Fresh"}).json()
    assert body["combined_label"] == "Fresh banana"
    assert body["produce"]["source"] == "user_selected"
    assert body["severity"] == {"status": "not_assessed", "label": None, "explanation": SEVERITY_TEXT}
    assert body["guidance"]["storage"]
    assert client.get("/api/guidance", params={"produce": "apple", "freshness": "Spoiled"}).json()["combined_label"] == "Spoiled apple"
    assert client.get("/api/guidance").json()["combined_label"] == "Fresh produce"
    bad = client.get("/api/guidance", params={"produce": "apple", "freshness": "Rotten"})
    assert bad.status_code == 400 and bad.json()["code"] == "unsupported_freshness"
    medium = client.get("/api/guidance", params={"produce": "apple", "freshness": "Medium"})
    assert medium.status_code == 400


def test_missing_model_and_invalid_image_still_behave(client: TestClient) -> None:
    ok_image = client.post(
        "/api/predict", data={"produce": "banana"}, files={"file": ("s.png", _png_bytes(), "image/png")}
    )
    assert ok_image.status_code == 503 and ok_image.json()["code"] == "model_unavailable"
    corrupt = client.post(
        "/api/predict", data={"produce": "banana"}, files={"file": ("n.png", b"not an image", "image/png")}
    )
    assert corrupt.status_code == 400 and corrupt.json()["code"] == "invalid_image"


# ------------------------------------------------------------------ real bundle


def test_fresh_banana_selected_by_user(ready_client: TestClient) -> None:
    data = SAMPLE.read_bytes()
    with_produce = ready_client.post(
        "/api/predict", data={"produce": "banana"}, files={"file": ("b.png", data, "image/png")}
    )
    assert with_produce.status_code == 200, with_produce.text
    body = with_produce.json()
    assert body["label"] == "Fresh"
    assert body["combined_label"] == "Fresh banana"
    assert body["produce"] == {
        "id": "banana",
        "label": "Banana",
        "source": "user_selected",
        "model_version": None,
        "probabilities": None,
    }
    assert body["freshness"]["label"] == "Fresh"
    assert body["freshness"]["class_index"] == 0
    assert body["freshness"]["model_version"] == body["model_version"]
    assert body["severity"]["status"] == "not_assessed" and body["severity"]["label"] is None
    assert body["severity"]["explanation"] == SEVERITY_TEXT
    assert body["guidance"]["storage"][0]["text"].startswith("Store bananas at room temperature")
    assert body["contract_version"] == "freshnet.predict.v2"

    from app.services.runtime import _bundle

    direct = classify_image(_bundle, SAMPLE)
    for name in ("Fresh", "Spoiled"):
        assert abs(body["probabilities"][name] - direct["probabilities"][name]) < 1e-5
        assert abs(body["freshness"]["probabilities"][name] - direct["probabilities"][name]) < 1e-5

    # The selection must not change the freshness result.
    without = ready_client.post("/api/predict", files={"file": ("b.png", data, "image/png")}).json()
    assert without["probabilities"] == body["probabilities"]
    assert without["label"] == body["label"]
    assert without["combined_label"] == "Fresh produce"
    assert without["produce"]["source"] == "unknown" and without["produce"]["id"] == "unknown"
    assert without["severity"]["status"] == "not_assessed"
    for key in ("label", "label_index", "probabilities", "model_version", "elapsed_ms", "timing_scope"):
        assert key in without


def test_spoiled_apple_selected_by_user(ready_client: TestClient) -> None:
    data = SPOILED_APPLE.read_bytes()
    response = ready_client.post(
        "/api/predict", data={"produce": "apple"}, files={"file": ("a.png", data, "image/png")}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["label"] == "Spoiled"
    assert body["combined_label"] == "Spoiled apple"
    assert body["produce"]["source"] == "user_selected"
    assert body["severity"]["status"] == "not_assessed"
    assert body["guidance"]["general_advice"][0]["text"] == "If it looks rotten, throw it away."
    generic = ready_client.post("/api/predict", files={"file": ("a.png", data, "image/png")}).json()
    assert generic["combined_label"] == "Spoiled produce"
    assert generic["probabilities"] == body["probabilities"]


def test_no_severity_or_forbidden_fields_in_response(ready_client: TestClient) -> None:
    body = ready_client.post(
        "/api/predict", data={"produce": "tomato"}, files={"file": ("a.png", SPOILED_APPLE.read_bytes(), "image/png")}
    ).json()
    text = json.dumps(body).lower()
    for phrase in ("percent_spoiled", "shelf_life", "expiry", "safe_to_eat", "mild", "moderate", "severe"):
        assert phrase not in text
    assert body["severity"] == {"status": "not_assessed", "label": None, "explanation": SEVERITY_TEXT}


# ------------------------------------------------------------------ future automatic identifier


class _FakeIdentifier:
    model_version = "fake-produce-v0"

    def identify(self, image: Any) -> ProduceIdentity:
        return ProduceIdentity("apple", "Apple", "model_predicted", self.model_version, {"apple": 0.9, "banana": 0.1})


class _BrokenIdentifier:
    model_version = "broken"

    def identify(self, image: Any) -> ProduceIdentity:
        raise RuntimeError("boom")


class _OutOfCatalogueIdentifier:
    model_version = "odd"

    def identify(self, image: Any) -> ProduceIdentity:
        return ProduceIdentity("dragonfruit", "Dragonfruit", "model_predicted")


@pytest.fixture
def reset_identifier():
    yield
    produce_service.set_identifier(None)


def test_future_identifier_is_reported_separately(ready_client: TestClient, reset_identifier) -> None:
    produce_service.set_identifier(_FakeIdentifier())
    data = SAMPLE.read_bytes()
    body = ready_client.post("/api/predict", files={"file": ("b.png", data, "image/png")}).json()
    assert body["produce"]["source"] == "model_predicted"
    assert body["produce"]["model_version"] == "fake-produce-v0"
    assert body["produce"]["probabilities"] == {"apple": 0.9, "banana": 0.1}
    assert body["model_version"] != body["produce"]["model_version"]
    assert body["combined_label"] == "Fresh apple"
    assert ready_client.get("/api/produce").json()["identification"]["available"] is True
    # A user selection always wins over the automatic identifier.
    selected = ready_client.post(
        "/api/predict", data={"produce": "banana"}, files={"file": ("b.png", data, "image/png")}
    ).json()
    assert selected["produce"]["source"] == "user_selected" and selected["combined_label"] == "Fresh banana"


@pytest.mark.parametrize("identifier", [_BrokenIdentifier(), _OutOfCatalogueIdentifier()])
def test_identifier_failure_keeps_freshness_and_unknown_identity(
    ready_client: TestClient, reset_identifier, identifier
) -> None:
    baseline = ready_client.post("/api/predict", files={"file": ("b.png", SAMPLE.read_bytes(), "image/png")}).json()
    produce_service.set_identifier(identifier)
    body = ready_client.post("/api/predict", files={"file": ("b.png", SAMPLE.read_bytes(), "image/png")}).json()
    assert body["label"] == "Fresh"
    assert body["probabilities"] == baseline["probabilities"]
    assert body["produce"]["source"] == "unknown" and body["produce"]["id"] == "unknown"
    assert body["combined_label"] == "Fresh produce"

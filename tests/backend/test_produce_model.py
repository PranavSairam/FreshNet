"""Automatic produce identification: bundle verification, API integration, fallbacks.

These tests build tiny *random-weight* bundles in a temporary directory. They check the
plumbing (verification, separate outputs, overrides, failure isolation), not accuracy.
Real produce accuracy comes only from the trained bundle's recorded evaluation. Real
freshness bundle checks reuse one train sample (Fresh banana) and one validation sample
(Rotten apple); the test split is not touched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.services import produce as produce_service
from app.services.produce_model import ProduceBundleError, load_produce_bundle, verify_produce_bundle
from produce.model import build_produce_model

ROOT = Path(__file__).resolve().parents[2]
FRESHNESS_BUNDLE = ROOT / "artifacts/deployment/v3/fused_epoch29_b962c650ed80"
SAMPLE = ROOT / "data/Dataset/Fresh/FreshBanana/b_f856.png"
SPOILED_APPLE = ROOT / "data/Dataset/Rotten/RottenApple/a_r596.png"
CLASS_ORDER = [
    "apple", "banana", "bitter_gourd", "capsicum", "carrot", "cucumber",
    "mango", "okra", "orange", "potato", "strawberry", "tomato",
]  # fmt: skip


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_bundle(
    directory: Path,
    *,
    confirmation_enabled: bool = False,
    threshold: float | None = None,
    smoke: bool = False,
    version: str = "freshnet-produce-test-random",
) -> Path:
    """Write a random-weight bundle in the produce bundle format."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = build_produce_model(len(CLASS_ORDER), {"dropout": 0.3}, pretrained=False)
    # A random-init EfficientNet collapses to a constant in eval mode. Calibrate the
    # BatchNorm statistics on noise so different images give different outputs.
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.momentum = None
    model.train()
    with torch.no_grad():
        for _ in range(2):
            model(torch.randn(8, 3, 224, 224))
    model.eval()
    torch.save({"state_dict": model.state_dict()}, directory / "weights.pt")
    (directory / "architecture.json").write_text(
        json.dumps(
            {
                "architecture": "efficientnet_b0",
                "num_classes": 12,
                "dropout": 0.3,
                "reconstruct_with_pretrained_download": False,
            }
        )
    )
    (directory / "class_mapping.json").write_text(
        json.dumps(
            {
                "class_order": CLASS_ORDER,
                "class_to_index": {name: i for i, name in enumerate(CLASS_ORDER)},
                "num_classes": 12,
            }
        )
    )
    (directory / "preprocessing.json").write_text(
        json.dumps(
            {
                "input_size": 224,
                "imagenet_mean": [0.485, 0.456, 0.406],
                "imagenet_std": [0.229, 0.224, 0.225],
                "eval_deterministic": True,
                "augmentation_at_inference": False,
            }
        )
    )
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "bundle_schema_version": "freshnet.produce.deployment.v1",
                "model_version": version,
                "smoke_test": smoke,
                "confirmation": {"enabled": confirmation_enabled, "threshold": threshold},
            }
        )
    )
    files = {
        name: _sha(directory / name)
        for name in ("weights.pt", "architecture.json", "class_mapping.json", "preprocessing.json", "metadata.json")
    }
    (directory / "checksums.json").write_text(json.dumps({"files": files}))
    return directory


def _start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, produce_bundle: Path) -> TestClient:
    monkeypatch.setenv("FRESHNET_SQLITE_PATH", str(tmp_path / "freshnet.sqlite"))
    monkeypatch.setenv("FRESHNET_BUNDLE_DIR", str(FRESHNESS_BUNDLE))
    monkeypatch.setenv("FRESHNET_PRODUCE_BUNDLE_DIR", str(produce_bundle))
    monkeypatch.setenv("FRESHNET_DEVICE", "cpu")
    get_settings.cache_clear()
    return TestClient(app)


@pytest.fixture
def produce_client(tmp_path, monkeypatch):
    bundle = make_bundle(tmp_path / "produce-bundle")
    with _start(tmp_path, monkeypatch, bundle) as test_client:
        yield test_client
    get_settings.cache_clear()
    produce_service.set_identifier(None)


def _predict(client: TestClient, path: Path, name: str = "image.png", **data: str) -> dict[str, Any]:
    response = client.post("/api/predict", data=data, files={"file": (name, path.read_bytes(), "image/png")})
    assert response.status_code == 200, response.text
    return response.json()


# ------------------------------------------------------------------ bundle verification


def test_valid_bundle_loads_and_is_reported(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path / "b")
    loaded = load_produce_bundle(bundle, device="cpu")
    assert loaded.class_order == CLASS_ORDER and loaded.model_version == "freshnet-produce-test-random"
    assert loaded.confirmation_enabled is False and loaded.confirmation_threshold is None


def test_tampered_weights_are_rejected(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path / "b")
    with (bundle / "weights.pt").open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(ProduceBundleError, match="Checksum mismatch"):
        verify_produce_bundle(bundle)


def test_smoke_bundle_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ProduceBundleError, match="smoke"):
        verify_produce_bundle(make_bundle(tmp_path / "b", smoke=True))


def test_enabled_confirmation_requires_threshold(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path / "b", confirmation_enabled=True, threshold=None)
    with pytest.raises(ProduceBundleError, match="threshold"):
        load_produce_bundle(bundle, device="cpu")


def test_missing_bundle_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ProduceBundleError):
        verify_produce_bundle(tmp_path / "nope")


# ------------------------------------------------------------------ API integration


def test_health_and_produce_endpoints_report_the_produce_model(produce_client: TestClient) -> None:
    health = produce_client.get("/api/health").json()
    assert health["model"]["model_version"] != health["produce_model"]["model_version"]
    assert health["produce_model"]["available"] is True and health["produce_model"]["status"] == "ready"
    assert health["produce_model"]["model_version"] == "freshnet-produce-test-random"
    info = produce_client.get("/api/produce").json()["identification"]
    assert info["mode"] == "automatic_with_correction" and info["available"] is True
    assert info["supported_ids"] == CLASS_ORDER


def test_both_models_run_and_outputs_stay_separate(produce_client: TestClient) -> None:
    body = _predict(produce_client, SAMPLE)
    produce = body["produce"]
    assert produce["source"] == "model_predicted" and produce["id"] in CLASS_ORDER
    assert produce["model_version"] == "freshnet-produce-test-random"
    assert set(produce["probabilities"]) == set(CLASS_ORDER)
    assert abs(sum(produce["probabilities"].values()) - 1.0) < 1e-4
    assert body["freshness"]["model_version"] == body["model_version"] != produce["model_version"]
    assert set(body["freshness"]["probabilities"]) == {"Fresh", "Spoiled"}
    noun = produce_service.load_catalogue()["produce"][produce["id"]]["noun"]
    assert body["combined_label"] == f"{body['label']} {noun}"
    assert body["identification"]["status"] == "model_predicted"
    assert body["severity"]["status"] == "not_assessed"


def test_freshness_result_is_unchanged_by_the_produce_model(produce_client: TestClient) -> None:
    with_model = _predict(produce_client, SAMPLE)
    produce_service.set_identifier(None)
    without = _predict(produce_client, SAMPLE)
    assert without["identification"]["status"] == "unavailable"
    assert without["probabilities"] == with_model["probabilities"]
    assert without["label"] == with_model["label"] and without["label_index"] == with_model["label_index"]
    assert without["freshness"] == with_model["freshness"]


def test_identification_comes_from_pixels_not_the_filename(produce_client: TestClient) -> None:
    first = _predict(produce_client, SAMPLE, name="banana.png")
    second = _predict(produce_client, SAMPLE, name="apple_fresh_tomato.jpg")
    assert first["produce"] == second["produce"]
    assert first["identification"] == second["identification"]


def test_different_images_can_get_different_identities(produce_client: TestClient) -> None:
    a = _predict(produce_client, SAMPLE)["produce"]["probabilities"]
    b = _predict(produce_client, SPOILED_APPLE)["produce"]["probabilities"]
    assert a != b  # the model reads the image, not a constant


def test_user_selection_overrides_and_skips_the_model(produce_client: TestClient) -> None:
    calls: list[int] = []
    real = produce_service._identifier

    class Spy:
        model_version = real.model_version
        class_order = real.class_order
        confirmation_threshold = None

        def identify(self, image: Any):
            calls.append(1)
            return real.identify(image)

    produce_service.set_identifier(Spy())
    body = _predict(produce_client, SAMPLE, produce="tomato")
    assert body["produce"]["source"] == "user_selected" and body["produce"]["id"] == "tomato"
    assert body["combined_label"] == f"{body['label']} tomato"
    assert body["identification"]["status"] == "user_selected"
    assert calls == []
    _predict(produce_client, SAMPLE)
    assert calls == [1]


def test_correcting_produce_after_a_result_does_not_run_any_model(produce_client: TestClient) -> None:
    calls: list[int] = []

    class Spy:
        model_version = "spy"
        class_order = CLASS_ORDER
        confirmation_threshold = None

        def identify(self, image: Any):
            calls.append(1)

    produce_service.set_identifier(Spy())
    body = produce_client.get("/api/guidance", params={"produce": "mango", "freshness": "Spoiled"}).json()
    assert body["combined_label"] == "Spoiled mango" and body["produce"]["source"] == "user_selected"
    assert calls == []


def test_produce_model_failure_keeps_freshness_working(produce_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = _predict(produce_client, SAMPLE)
    import app.services.produce_model as pm

    def boom(*_args: Any, **_kwargs: Any):
        raise RuntimeError("produce model exploded")

    monkeypatch.setattr(pm, "predict_probabilities", boom)
    body = _predict(produce_client, SAMPLE)
    assert body["identification"]["status"] == "failed"
    assert body["produce"]["source"] == "unknown" and body["produce"]["id"] == "unknown"
    assert body["combined_label"] == f"{baseline['label']} produce"
    assert body["probabilities"] == baseline["probabilities"] and body["label"] == baseline["label"]
    assert body["guidance"]["review_status"] == "generic"


def test_missing_produce_bundle_falls_back_to_manual_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with _start(tmp_path, monkeypatch, tmp_path / "absent") as test_client:
        assert test_client.get("/api/health").json()["produce_model"]["available"] is False
        body = _predict(test_client, SAMPLE)
        assert body["identification"]["status"] == "unavailable" and body["produce"]["id"] == "unknown"
        chosen = _predict(test_client, SAMPLE, produce="banana")
        assert chosen["combined_label"] == f"{chosen['label']} banana"
    get_settings.cache_clear()


def test_corrupt_produce_bundle_does_not_break_startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = make_bundle(tmp_path / "b")
    (bundle / "weights.pt").write_bytes(b"corrupt")
    with _start(tmp_path, monkeypatch, bundle) as test_client:
        assert test_client.get("/api/health").json()["produce_model"]["available"] is False
        assert _predict(test_client, SAMPLE)["identification"]["status"] == "unavailable"
    get_settings.cache_clear()


# ------------------------------------------------------------------ uncertainty


def test_low_confidence_asks_for_confirmation_without_claiming_an_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = make_bundle(tmp_path / "b", confirmation_enabled=True, threshold=0.999)
    with _start(tmp_path, monkeypatch, bundle) as test_client:
        body = _predict(test_client, SAMPLE)
        ident = body["identification"]
        assert ident["status"] == "needs_confirmation" and ident["confirmation_required"] is True
        assert ident["suggestion"]["id"] in CLASS_ORDER and 0 < ident["suggestion"]["confidence"] < 0.999
        assert ident["confirmation_threshold"] == 0.999
        assert body["produce"]["id"] == "unknown" and body["produce"]["source"] == "unknown"
        assert body["combined_label"] == f"{body['label']} produce"
        assert set(ident["probabilities"]) == set(CLASS_ORDER)
        # Confirming (user selection) resolves it without the model.
        confirmed = _predict(test_client, SAMPLE, produce=ident["suggestion"]["id"])
        assert confirmed["produce"]["source"] == "user_selected"
    get_settings.cache_clear()


def test_confident_prediction_is_not_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = make_bundle(tmp_path / "b", confirmation_enabled=True, threshold=0.0)
    with _start(tmp_path, monkeypatch, bundle) as test_client:
        body = _predict(test_client, SAMPLE)
        assert body["identification"]["status"] == "model_predicted"
        assert body["identification"]["confirmation_required"] is False
    get_settings.cache_clear()


def test_identifier_error_text_never_leaks_into_the_response(produce_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.services.produce_model as pm

    def boom(*_args: Any, **_kwargs: Any):
        raise RuntimeError("secret-internal-detail")

    monkeypatch.setattr(pm, "predict_probabilities", boom)
    text = json.dumps(_predict(produce_client, SAMPLE))
    assert "secret-internal-detail" not in text


# ------------------------------------------------------------------ the real trained bundle

REAL_PRODUCE_BUNDLE = ROOT / "artifacts/deployment/produce_v1/effb0_epoch10_e389c317c8e5"
# Validation-split images only (never the test split), one per produce type, with their true type.
REAL_SAMPLES = [
    ("data/Dataset/Fresh/FreshApple/a_f622.png", "apple"),
    ("data/Dataset/Fresh/FreshBanana/b_f856.png", "banana"),
    ("data/Dataset/Fresh/FreshTomato/tomato1_26.jpg_0_7135.jpg", "tomato"),
    ("data/Dataset/Rotten/RottenCarrot/rottenCarrot (402).jpg", "carrot"),
    ("data/Dataset/Fresh/FreshOkara/o_f561.png", "okra"),
    ("data/Dataset/Rotten/RottenMango/rottenMango (565).jpg", "mango"),
]
needs_real_bundle = pytest.mark.skipif(not REAL_PRODUCE_BUNDLE.is_dir(), reason="trained produce bundle not present")


@needs_real_bundle
def test_real_bundle_verifies_and_is_not_a_smoke_artifact() -> None:
    checked = verify_produce_bundle(REAL_PRODUCE_BUNDLE)
    meta = checked["metadata"]
    assert meta["smoke_test"] is False and meta["model_version"] == "freshnet-produce-v1-effb0-e389c317c8e5"
    assert checked["class_mapping"]["class_order"] == CLASS_ORDER
    assert meta["manifest_fingerprint"] == "9e3a27855e6e01d97a32fee61273dde59b1df7e3166ef25e764e03b97c872c66"
    assert meta["test_metrics"]["scope"].startswith("produce test subset")


@needs_real_bundle
def test_real_bundle_identifies_validation_images_through_the_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with _start(tmp_path, monkeypatch, REAL_PRODUCE_BUNDLE) as test_client:
        assert test_client.get("/api/health").json()["produce_model"]["status"] == "ready"
        for relative, expected in REAL_SAMPLES:
            # A misleading filename must not matter.
            body = _predict(test_client, ROOT / relative, name="not_the_produce.png")
            identification = body["identification"]
            assert identification["status"] in {"model_predicted", "needs_confirmation"}, relative
            best = max(identification["probabilities"], key=identification["probabilities"].get)
            assert best == expected, (relative, best)
            if identification["status"] == "model_predicted":
                assert body["produce"]["id"] == expected
                noun = produce_service.load_catalogue()["produce"][expected]["noun"]
                assert body["combined_label"].endswith(noun)
            assert body["label"] in {"Fresh", "Spoiled"}
    get_settings.cache_clear()

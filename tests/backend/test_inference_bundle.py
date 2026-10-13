import json
from pathlib import Path

import pytest
import torch

from app.services.inference import BundleError, classify_image, load_bundle, verify_bundle_files
from models import build_model


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _tiny_bundle(directory: Path, *, fresh: int = 0, spoiled: int = 1, smoke: bool = False) -> None:
    directory.mkdir(parents=True)
    model = build_model(
        "fused",
        {
            "embedding_dim": 32,
            "dropout": 0.0,
            "context": {"pretrained": False},
            "detail": {"channels": [4, 8]},
        },
        num_classes=2,
    )
    torch.save({"state_dict": model.state_dict()}, directory / "weights.pt")
    _write(
        directory / "architecture.json",
        {
            "architecture": "fused",
            "embedding_dim": 32,
            "dropout": 0.0,
            "reconstruct_with_pretrained_download": False,
            "detail": {"channels": [4, 8], "patch_layout": "corners_center"},
        },
    )
    _write(
        directory / "class_mapping.json",
        {"Fresh": fresh, "Spoiled": spoiled, "num_classes": 2, "label_order": ["Fresh", "Spoiled"]},
    )
    _write(
        directory / "preprocessing.json",
        {
            "context_size": 224,
            "detail_size": 384,
            "patch_size": 160,
            "imagenet_mean": [0.485, 0.456, 0.406],
            "imagenet_std": [0.229, 0.224, 0.225],
            "eval_deterministic": True,
            "augmentation_at_inference": False,
        },
    )
    _write(
        directory / "metadata.json",
        {
            "bundle_schema_version": "freshnet.deployment.v1",
            "model_version": "test-bundle",
            "architecture": "fused",
            "smoke_test": smoke,
        },
    )
    files = {
        name: _sha256(directory / name)
        for name in ("weights.pt", "architecture.json", "class_mapping.json", "preprocessing.json", "metadata.json")
    }
    _write(directory / "checksums.json", {"algorithm": "sha256", "model_version": "test-bundle", "files": files})


def test_tiny_bundle_loads_and_classifies(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    _tiny_bundle(bundle_dir)
    image = tmp_path / "sample.png"
    torch.manual_seed(0)
    from PIL import Image

    Image.fromarray(torch.randint(0, 255, (80, 70, 3), dtype=torch.uint8).numpy(), mode="RGB").save(image)
    loaded = load_bundle(bundle_dir, device="cpu")
    result = classify_image(loaded, image)
    assert result["model_version"] == "test-bundle"
    assert result["label"] in {"Fresh", "Spoiled"}
    assert result["decision_rule"] == "argmax"
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-5
    assert loaded.class_mapping == {"Fresh": 0, "Spoiled": 1}


def test_corrupt_weights_fail(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    _tiny_bundle(bundle_dir)
    weights = bundle_dir / "weights.pt"
    weights.write_bytes(weights.read_bytes() + b"corrupt")
    with pytest.raises(BundleError, match="Checksum mismatch"):
        verify_bundle_files(bundle_dir)


def test_incompatible_labels_fail(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    _tiny_bundle(bundle_dir, fresh=1, spoiled=0)
    with pytest.raises(BundleError, match="class mapping"):
        verify_bundle_files(bundle_dir)


def test_smoke_artifact_fails(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    _tiny_bundle(bundle_dir, smoke=True)
    with pytest.raises(BundleError, match="smoke"):
        verify_bundle_files(bundle_dir)


def test_missing_bundle_fails(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="not found"):
        load_bundle(tmp_path / "missing", device="cpu")

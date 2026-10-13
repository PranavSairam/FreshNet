from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from dataset.integrity import ManifestRow, find_cross_split_groups, stable_split_group_key
from models import ContextOnlyFreshNet, DetailOnlyFreshNet, FusedFreshNet, build_model
from training.config import validate_num_classes
from training.data import ManifestFreshnessDataset, extract_corner_center_patches, preprocessing_dict
from training.metrics import compute_classification_metrics
from PIL import Image


def test_stable_group_key_binds_exact_hashes():
    a = ManifestRow("a.png", "Fresh", "train", "heuristic:x", "abc", "", "")
    b = ManifestRow("b.png", "Fresh", "val", "heuristic:y", "abc", "near:foo", "")
    assert stable_split_group_key(a) == stable_split_group_key(b) == "exact:abc"


def test_cross_split_detector_finds_exact_leak():
    rows = [
        ManifestRow("a.png", "Fresh", "train", "g", "abc", "", ""),
        ManifestRow("b.png", "Fresh", "test", "g", "abc", "", ""),
    ]
    leaks = find_cross_split_groups(rows, key_fn=lambda r: f"exact:{r.content_sha256}")
    assert len(leaks) == 1


def test_binary_mapping_validation():
    config = {
        "task": {"num_classes": 2, "labels": {"Fresh": 0, "Spoiled": 1}},
    }
    mapping = {"Fresh": 0, "Spoiled": 1, "num_classes": 2}
    assert validate_num_classes(config, mapping) == 2
    with pytest.raises(ValueError):
        validate_num_classes(config, {"Fresh": 0, "Spoiled": 1, "num_classes": 3})


def test_model_output_shapes_and_gradients():
    context = torch.randn(2, 3, 224, 224)
    patches = torch.randn(2, 5, 3, 160, 160)
    for arch, model in [
        ("context", ContextOnlyFreshNet(num_classes=2, pretrained=False)),
        ("detail", DetailOnlyFreshNet(num_classes=2)),
        ("fused", FusedFreshNet(num_classes=2, pretrained=False)),
    ]:
        if arch == "context":
            logits = model(context=context)
        elif arch == "detail":
            logits = model(patches=patches)
        else:
            logits = model(context=context, patches=patches)
        assert logits.shape == (2, 2)
        loss = logits.sum()
        loss.backward()
        grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
        assert grads and all(grads)


def test_build_model_rejects_three_classes():
    with pytest.raises(ValueError):
        build_model("fused", {"embedding_dim": 32, "dropout": 0.1}, num_classes=3)


def test_checkpoint_roundtrip(tmp_path: Path):
    model = FusedFreshNet(num_classes=2, embedding_dim=32, pretrained=False)
    path = tmp_path / "ckpt.pt"
    payload = {
        "architecture": "fused",
        "state_dict": model.state_dict(),
        "num_classes": 2,
        "class_mapping": {"Fresh": 0, "Spoiled": 1, "num_classes": 2},
        "preprocessing": {"context_size": 224, "detail_size": 384, "patch_size": 160},
        "smoke_test": True,
    }
    torch.save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    model2 = FusedFreshNet(num_classes=2, embedding_dim=32, pretrained=False)
    model2.load_state_dict(loaded["state_dict"])
    model.eval()
    model2.eval()
    context = torch.randn(1, 3, 224, 224)
    patches = torch.randn(1, 5, 3, 160, 160)
    with torch.no_grad():
        a = model(context=context, patches=patches)
        b = model2(context=context, patches=patches)
    assert torch.allclose(a, b, atol=1e-5)


def test_patch_extraction_and_metrics():
    img = Image.new("RGB", (400, 400), (10, 20, 30))
    patches = extract_corner_center_patches(img, patch_size=160)
    assert len(patches) == 5
    metrics = compute_classification_metrics([0, 1, 1, 0], [0, 0, 1, 0])
    assert metrics["spoiled_to_fresh_errors"] == 1
    assert metrics["spoiled_test_images"] == 2
    assert metrics["spoiled_to_fresh_rate"] == 0.5


def test_manifest_dataset_binary_labels(tmp_path: Path):
    # Minimal synthetic images + CSV
    img_a = tmp_path / "a.png"
    img_b = tmp_path / "b.png"
    Image.new("RGB", (200, 200), (0, 200, 0)).save(img_a)
    Image.new("RGB", (200, 200), (200, 0, 0)).save(img_b)
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "image_path,label,fixture\n"
        f"{img_a},Fresh,false\n"
        f"{img_b},Spoiled,false\n",
        encoding="utf-8",
    )
    config = {
        "preprocessing": {
            "context_size": 224,
            "detail_size": 384,
            "patch_size": 160,
            "imagenet_mean": [0.485, 0.456, 0.406],
            "imagenet_std": [0.229, 0.224, 0.225],
            "train_augmentation": {
                "horizontal_flip_p": 0.0,
                "rotation_degrees": 0,
                "color_jitter": {"brightness": 0, "contrast": 0, "saturation": 0, "hue": 0},
            },
            "eval_deterministic": True,
        }
    }
    ds = ManifestFreshnessDataset(
        csv_path,
        config,
        {"Fresh": 0, "Spoiled": 1},
        train=False,
    )
    item0 = ds[0]
    item1 = ds[1]
    assert item0["context"].shape == (3, 224, 224)
    assert item0["patches"].shape == (5, 3, 160, 160)
    assert item0["label"] == 0
    assert item1["label"] == 1
    # Deterministic eval transform: repeated read matches.
    again = ds[0]
    assert torch.allclose(item0["context"], again["context"])

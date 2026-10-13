from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from dataset.build_fixture import FIXTURE_ROOT, main as build_fixture
from dataset.config import DatasetConfig, load_config
from dataset.duplicates import detect_exact_duplicates, file_sha256
from dataset.ingest import ingest_class_folders, ingest_csv_manifest
from dataset.labels import map_label, resolve_labels
from dataset.prepare import run_preparation
from dataset.split import audit_official_splits, create_group_aware_splits
from dataset.transforms import build_eval_transform_spec, build_train_transform_spec
from dataset.validate import validate_records


@pytest.fixture(scope="module")
def fixture_root() -> Path:
    build_fixture()
    assert FIXTURE_ROOT.is_dir()
    return FIXTURE_ROOT


def _config_from_dict(data: dict, tmp_path: Path) -> DatasetConfig:
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_config(path)


def test_medium_absent_is_explained_and_not_invented():
    resolution = resolve_labels(
        ["Fresh", "Fresh", "Rotten", "Spoiled"],
        {"Fresh": "Fresh", "Rotten": "Spoiled", "Spoiled": "Spoiled", "Medium": "Medium"},
    )
    assert resolution.has_medium is False
    assert resolution.three_class_ready is False
    assert resolution.two_class_ready is True
    assert resolution.canonical_labels == ["Fresh", "Spoiled"]
    assert any("Medium is absent" in note for note in resolution.limitation_notes)
    assert "Medium" not in resolution.mapped_counts


def test_map_label_is_case_insensitive():
    mapping = {"Fresh": "Fresh", "rotten": "Spoiled"}
    assert map_label("fresh", mapping) == "Fresh"
    assert map_label("Rotten", mapping) == "Spoiled"
    assert map_label("Medium", mapping) is None


def test_class_folder_ingest_and_validation(fixture_root: Path, tmp_path: Path):
    base = load_config().raw
    base["dataset"]["source_path"] = str(fixture_root / "class_folders")
    base["dataset"]["input_mode"] = "class_folders"
    base["duplicates"]["enable_near"] = True
    base["duplicates"]["near_max_images"] = 100
    config = _config_from_dict(base, tmp_path)

    records = ingest_class_folders(config)
    assert len(records) >= 5
    assert {r.label_raw for r in records} == {"Fresh", "Rotten"}
    summary = validate_records(records, config)
    assert summary["unreadable"] >= 1
    assert summary["readable"] >= 4


def test_exact_duplicate_detection(fixture_root: Path):
    a = fixture_root / "class_folders" / "Fresh" / "Banana" / "specimen2.png"
    b = fixture_root / "class_folders" / "Fresh" / "Banana" / "specimen2_copy.png"
    assert file_sha256(a) == file_sha256(b)

    from dataset.records import ImageRecord

    records = [
        ImageRecord(image_path=str(a), label_raw="Fresh", label="Fresh", readable=True),
        ImageRecord(image_path=str(b), label_raw="Fresh", label="Fresh", readable=True),
    ]
    summary = detect_exact_duplicates(records)
    assert summary["exact_duplicate_groups"] == 1
    assert records[1].exact_duplicate_of == records[0].image_path


def test_csv_ingest_and_group_split_no_leakage(fixture_root: Path, tmp_path: Path):
    base = load_config().raw
    base["dataset"]["input_mode"] = "csv_manifest"
    base["dataset"]["csv_manifest_path"] = str(fixture_root / "csv_mode" / "manifest.csv")
    base["dataset"]["source_path"] = str(fixture_root / "csv_mode")
    base["output"]["processed_dir"] = str(tmp_path / "processed")
    base["output"]["manifests_dir"] = str(tmp_path / "processed" / "manifests")
    base["output"]["reports_dir"] = str(tmp_path / "processed" / "reports")
    config = _config_from_dict(base, tmp_path)

    records = ingest_csv_manifest(config)
    assert any(r.specimen_id == "specA" for r in records)
    validate_records(records, config)
    split = create_group_aware_splits(records, config)

    # All images of specimen specA must share one split.
    spec_a_splits = {
        split.assignments[r.image_path]
        for r in records
        if r.specimen_id == "specA" and r.image_path in split.assignments
    }
    assert len(spec_a_splits) == 1


def test_official_split_audit_detects_group_leakage(tmp_path: Path):
    from dataset.records import ImageRecord

    records = [
        ImageRecord(
            image_path=str(tmp_path / "a.png"),
            label_raw="Fresh",
            label="Fresh",
            specimen_id="same",
            split_official="train",
            group_id="specimen:same",
            readable=True,
        ),
        ImageRecord(
            image_path=str(tmp_path / "b.png"),
            label_raw="Fresh",
            label="Fresh",
            specimen_id="same",
            split_official="test",
            group_id="specimen:same",
            readable=True,
        ),
    ]
    # Create dummy files so usability filters pass if needed.
    for record in records:
        Path(record.image_path).write_bytes(b"x")
    audit = audit_official_splits(records)
    assert audit["present"] is True
    assert audit["groups_spanning_splits"] == 1


def test_transforms_train_only_mild_color():
    raw = load_config().raw
    train = build_train_transform_spec(raw)
    eval_spec = build_eval_transform_spec(raw)
    assert train["augmentation_scope"] == "training_only"
    assert eval_spec["deterministic"] is True
    jitter = next(op for op in train["operations"] if op["name"] == "color_jitter")
    assert jitter["params"]["hue"] <= 0.02
    assert jitter["params"]["saturation"] <= 0.1


def test_prepare_end_to_end_on_fixture(fixture_root: Path, tmp_path: Path):
    base = load_config().raw
    base["dataset"]["source_path"] = str(fixture_root / "class_folders")
    base["dataset"]["input_mode"] = "class_folders"
    base["duplicates"]["enable_near"] = True
    base["duplicates"]["near_max_images"] = 50
    base["output"]["processed_dir"] = str(tmp_path / "processed")
    base["output"]["manifests_dir"] = str(tmp_path / "processed" / "manifests")
    base["output"]["reports_dir"] = str(tmp_path / "processed" / "reports")
    config = _config_from_dict(base, tmp_path)

    result = run_preparation(config)
    assert result["label_resolution"].three_class_ready is False
    assert Path(result["report_path"]).is_file()
    assert Path(result["manifest_paths"]["train"]).is_file()
    assert Path(result["manifest_paths"]["fingerprint"]).is_file()

    fingerprint = json.loads(Path(result["manifest_paths"]["fingerprint"]).read_text(encoding="utf-8"))
    assert fingerprint["split_fingerprint"] == result["fingerprint"]

    # Re-run must produce identical fingerprint for same inputs.
    result2 = run_preparation(config)
    assert result2["fingerprint"] == result["fingerprint"]

    with Path(result["manifest_paths"]["all"]).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(row["label"] in {"Fresh", "Spoiled"} for row in rows)
    assert all(row["label"] != "Medium" for row in rows)

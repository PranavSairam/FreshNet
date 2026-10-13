from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG_PATH, DatasetConfig, load_config
from .duplicates import run_duplicate_detection
from .ingest import ingest
from .labels import resolve_labels
from .records import ImageRecord
from .report import write_audit_report
from .split import create_group_aware_splits
from .transforms import try_build_torchvision_transforms
from .validate import validate_records

MANIFEST_FIELDS = [
    "image_path",
    "label",
    "label_raw",
    "food_type",
    "specimen_id",
    "capture_session",
    "source",
    "split",
    "group_id",
    "width",
    "height",
    "readable",
    "content_sha256",
    "phash",
    "exact_duplicate_of",
    "near_duplicate_group",
    "problems",
    "fixture",
]


def write_manifests(
    records: list[ImageRecord],
    assignments: dict[str, str],
    manifests_dir: Path,
    fingerprint: str,
    label_resolution: Any,
) -> dict[str, Path]:
    manifests_dir.mkdir(parents=True, exist_ok=True)
    usable = [
        r
        for r in records
        if r.image_path in assignments
        and not r.fixture
        and r.readable
        and r.label is not None
        and r.label in label_resolution.canonical_labels
    ]

    paths: dict[str, Path] = {}
    by_split: dict[str, list[ImageRecord]] = {"train": [], "val": [], "test": [], "all": usable}
    for record in usable:
        by_split[assignments[record.image_path]].append(record)

    for split_name, items in by_split.items():
        path = manifests_dir / f"{split_name}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for record in items:
                row = {
                    "image_path": record.image_path,
                    "label": record.label,
                    "label_raw": record.label_raw,
                    "food_type": record.food_type or "",
                    "specimen_id": record.specimen_id or "",
                    "capture_session": record.capture_session or "",
                    "source": record.source or "",
                    "split": assignments.get(record.image_path, ""),
                    "group_id": record.group_id or "",
                    "width": record.width if record.width is not None else "",
                    "height": record.height if record.height is not None else "",
                    "readable": record.readable,
                    "content_sha256": record.content_sha256 or "",
                    "phash": record.phash or "",
                    "exact_duplicate_of": record.exact_duplicate_of or "",
                    "near_duplicate_group": record.near_duplicate_group or "",
                    "problems": "|".join(record.problems),
                    "fixture": record.fixture,
                }
                writer.writerow(row)
        paths[split_name] = path

    problems_path = manifests_dir / "problems.csv"
    with problems_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            if not record.problems:
                continue
            writer.writerow(
                {
                    "image_path": record.image_path,
                    "label": record.label or "",
                    "label_raw": record.label_raw,
                    "food_type": record.food_type or "",
                    "specimen_id": record.specimen_id or "",
                    "capture_session": record.capture_session or "",
                    "source": record.source or "",
                    "split": assignments.get(record.image_path, ""),
                    "group_id": record.group_id or "",
                    "width": record.width if record.width is not None else "",
                    "height": record.height if record.height is not None else "",
                    "readable": record.readable,
                    "content_sha256": record.content_sha256 or "",
                    "phash": record.phash or "",
                    "exact_duplicate_of": record.exact_duplicate_of or "",
                    "near_duplicate_group": record.near_duplicate_group or "",
                    "problems": "|".join(record.problems),
                    "fixture": record.fixture,
                }
            )
    paths["problems"] = problems_path

    fingerprint_path = manifests_dir / "split_fingerprint.json"
    fingerprint_path.write_text(
        json.dumps(
            {
                "split_fingerprint": fingerprint,
                "canonical_labels": label_resolution.canonical_labels,
                "three_class_ready": label_resolution.three_class_ready,
                "manifests": {k: str(v) for k, v in paths.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["fingerprint"] = fingerprint_path
    return paths


def run_preparation(
    config: DatasetConfig,
    *,
    skip_near_duplicates: bool = False,
    max_images: int | None = None,
) -> dict[str, Any]:
    records = ingest(config)
    if max_images is not None:
        records = records[: max(0, max_images)]

    # Exclude synthetic fixtures from real training manifests later; keep them for audit visibility.
    raw_labels = [r.label_raw for r in records if not r.fixture]
    label_resolution = resolve_labels(
        raw_labels,
        config.label_mapping,
        intended_classes=config.intended_classes,
        require_for_three_class=config.require_for_three_class,
        allow_two_class_training=config.allow_two_class_training,
    )

    # Drop unmapped from training by leaving label None already.
    validation_summary = validate_records(records, config)
    duplicate_summary = run_duplicate_detection(
        records,
        config,
        enable_near=None if not skip_near_duplicates else False,
    )
    split_result = create_group_aware_splits(records, config)
    transform_specs = try_build_torchvision_transforms(config.raw)

    config.manifests_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths = write_manifests(
        records,
        split_result.assignments,
        config.manifests_dir,
        split_result.fingerprint,
        label_resolution,
    )
    report_path = write_audit_report(
        config=config,
        records=records,
        label_resolution=label_resolution,
        validation_summary=validation_summary,
        duplicate_summary=duplicate_summary,
        split_summary=split_result.summary,
        leakage_warnings=split_result.leakage_warnings,
        transform_specs=transform_specs,
        fingerprint=split_result.fingerprint,
        output_path=config.reports_dir / "dataset_audit.json",
    )

    # Persist transform specs for experiment reuse without requiring torch at prep time.
    transforms_path = config.processed_dir / "transforms.json"
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    transforms_path.write_text(
        json.dumps(
            {
                "train": transform_specs.get("train_spec"),
                "eval": transform_specs.get("eval_spec"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "records": len(records),
        "label_resolution": label_resolution,
        "validation_summary": validation_summary,
        "duplicate_summary": duplicate_summary,
        "split_summary": split_result.summary,
        "fingerprint": split_result.fingerprint,
        "manifest_paths": {k: str(v) for k, v in manifest_paths.items()},
        "report_path": str(report_path),
        "transforms_path": str(transforms_path),
        "leakage_warnings": split_result.leakage_warnings,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FreshNet dataset manifests and audit report.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to configs/dataset.yaml")
    parser.add_argument(
        "--source",
        default=None,
        help="Override dataset.source_path (class-folder root or ignored in csv mode).",
    )
    parser.add_argument(
        "--input-mode",
        choices=["class_folders", "csv_manifest"],
        default=None,
        help="Override dataset.input_mode",
    )
    parser.add_argument("--csv", default=None, help="Override dataset.csv_manifest_path")
    parser.add_argument("--skip-near-duplicates", action="store_true")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for smoke runs. Do not use for final experiment manifests.",
    )
    parser.add_argument(
        "--require-three-class",
        action="store_true",
        help="Exit non-zero if Medium is missing / three-class training is not ready.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(Path(args.config))
    if args.source:
        config.raw["dataset"]["source_path"] = args.source
    if args.input_mode:
        config.raw["dataset"]["input_mode"] = args.input_mode
    if args.csv:
        config.raw["dataset"]["csv_manifest_path"] = args.csv
        config.raw["dataset"]["input_mode"] = "csv_manifest"

    try:
        result = run_preparation(
            config,
            skip_near_duplicates=args.skip_near_duplicates,
            max_images=args.max_images,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "Place the dataset under data/Dataset (or set --source / configs/dataset.yaml).",
            file=sys.stderr,
        )
        return 2

    labels = result["label_resolution"]
    print(json.dumps(
        {
            "records": result["records"],
            "canonical_labels": labels.canonical_labels,
            "mapped_counts": labels.mapped_counts,
            "three_class_ready": labels.three_class_ready,
            "split_fingerprint": result["fingerprint"],
            "report_path": result["report_path"],
            "manifests": result["manifest_paths"],
            "limitation_notes": labels.limitation_notes,
            "leakage_warnings": result["leakage_warnings"],
        },
        indent=2,
    ))

    if args.require_three_class and not labels.three_class_ready:
        print("Three-class requirement not met. Medium was not manufactured.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

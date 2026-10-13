from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DatasetConfig, deep_get
from .labels import LabelResolution
from .records import ImageRecord


def find_license_text(source_path: Path, candidates: list[str]) -> dict[str, Any]:
    found = []
    for name in candidates:
        for path in source_path.rglob(name):
            if path.is_file() and path.stat().st_size < 200_000:
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                found.append(
                    {
                        "path": str(path),
                        "preview": text[:500],
                    }
                )
                if len(found) >= 5:
                    break
        if len(found) >= 5:
            break
    return {
        "files_found": found,
        "note": (
            "No license claim is invented by FreshNet. If no license file is found, "
            "record provenance manually before sharing the dataset."
        ),
    }


def write_audit_report(
    *,
    config: DatasetConfig,
    records: list[ImageRecord],
    label_resolution: LabelResolution,
    validation_summary: dict[str, Any],
    duplicate_summary: dict[str, Any],
    split_summary: dict[str, Any],
    leakage_warnings: list[str],
    transform_specs: dict[str, Any],
    fingerprint: str,
    output_path: Path,
) -> Path:
    license_info = find_license_text(
        config.source_path if config.source_path.exists() else config.path.parent,
        list(deep_get(config, "dataset", "license_candidates", default=[])),
    )
    fixture_count = sum(1 for r in records if r.fixture)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": deep_get(config, "dataset", "name"),
        "source_path": str(config.source_path),
        "input_mode": config.input_mode,
        "provenance": {
            "note": str(deep_get(config, "dataset", "provenance_note", default="")).strip(),
            "license": license_info,
            "download_url_claimed_by_project": None,
        },
        "counts": {
            "total_ingested": len(records),
            "readable": validation_summary.get("readable"),
            "missing_files": validation_summary.get("missing_files"),
            "unreadable": validation_summary.get("unreadable"),
            "invalid_labels": validation_summary.get("invalid_labels"),
            "fixture_records_excluded_from_real_training": fixture_count,
        },
        "labels": {
            "intended_classes": config.intended_classes,
            "raw_label_counts": label_resolution.raw_label_counts,
            "mapped_counts": label_resolution.mapped_counts,
            "canonical_training_labels": label_resolution.canonical_labels,
            "has_fresh": label_resolution.has_fresh,
            "has_medium": label_resolution.has_medium,
            "has_spoiled": label_resolution.has_spoiled,
            "three_class_ready": label_resolution.three_class_ready,
            "two_class_ready": label_resolution.two_class_ready,
            "unmapped_labels": label_resolution.unmapped_labels,
            "limitation_notes": label_resolution.limitation_notes,
        },
        "class_imbalance": validation_summary.get("class_imbalance"),
        "dimension_stats": validation_summary.get("dimension_stats"),
        "duplicates": duplicate_summary,
        "splits": split_summary,
        "leakage_warnings": leakage_warnings,
        "transforms": {
            "train": transform_specs.get("train_spec"),
            "eval": transform_specs.get("eval_spec"),
            "torchvision_available": transform_specs.get("torchvision_available"),
        },
        "split_fingerprint": fingerprint,
        "problems_index": _problem_index(records),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_path = output_path.with_suffix(".md")
    md_path.write_text(_to_markdown(report), encoding="utf-8")
    return output_path


def _problem_index(records: list[ImageRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for problem in record.problems:
            counts[problem] = counts.get(problem, 0) + 1
    return dict(sorted(counts.items()))


def _to_markdown(report: dict[str, Any]) -> str:
    labels = report["labels"]
    counts = report["counts"]
    lines = [
        "# FreshNet dataset audit",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Source: `{report['source_path']}`",
        f"Input mode: `{report['input_mode']}`",
        f"Split fingerprint: `{report['split_fingerprint']}`",
        "",
        "## Counts",
        "",
        f"- Total ingested: **{counts['total_ingested']}**",
        f"- Readable: **{counts['readable']}**",
        f"- Missing files: **{counts['missing_files']}**",
        f"- Unreadable: **{counts['unreadable']}**",
        f"- Invalid labels: **{counts['invalid_labels']}**",
        "",
        "## Labels",
        "",
        f"- Intended: {labels['intended_classes']}",
        f"- Mapped counts: `{labels['mapped_counts']}`",
        f"- Canonical training labels: **{labels['canonical_training_labels']}**",
        f"- Fresh present: {labels['has_fresh']}",
        f"- Medium present: {labels['has_medium']}",
        f"- Spoiled present: {labels['has_spoiled']}",
        f"- Three-class ready: **{labels['three_class_ready']}**",
        "",
    ]
    for note in labels.get("limitation_notes", []):
        lines.append(f"- Note: {note}")
    lines.extend(
        [
            "",
            "## Leakage warnings",
            "",
        ]
    )
    warnings = report.get("leakage_warnings") or ["None recorded."]
    for warning in warnings:
        lines.append(f"- {warning}")
    lines.extend(
        [
            "",
            "## Provenance / license",
            "",
            report["provenance"]["note"],
            "",
            report["provenance"]["license"]["note"],
            "",
            "This project does not invent a dataset download URL.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"

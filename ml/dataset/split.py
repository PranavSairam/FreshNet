from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .config import DatasetConfig, deep_get
from .records import ImageRecord


@dataclass
class SplitResult:
    assignments: dict[str, str]  # image_path -> split
    mode: str
    fingerprint: str
    summary: dict[str, Any]
    leakage_warnings: list[str]


def _group_key(record: ImageRecord) -> str:
    # Exact duplicates always share a split. Near-duplicate links do not.
    if record.content_sha256:
        return f"exact:{record.content_sha256}"
    base = (record.group_id or record.image_path).split("|near:")[0]
    return base


def audit_official_splits(records: list[ImageRecord]) -> dict[str, Any]:
    with_official = [r for r in records if r.split_official in {"train", "val", "test"}]
    if not with_official:
        return {"present": False, "groups_spanning_splits": 0, "images_with_official_split": 0}

    group_to_splits: dict[str, set[str]] = defaultdict(set)
    for record in with_official:
        group_to_splits[_group_key(record)].add(record.split_official or "unknown")

    spanning = {group: sorted(splits) for group, splits in group_to_splits.items() if len(splits) > 1}
    return {
        "present": True,
        "images_with_official_split": len(with_official),
        "groups_spanning_splits": len(spanning),
        "example_spanning_groups": dict(list(spanning.items())[:20]),
        "split_counts": dict(Counter(r.split_official for r in with_official)),
    }


def _majority_label(items: list[ImageRecord]) -> str:
    counts = Counter(r.label for r in items if r.label)
    if not counts:
        return "unknown"
    return counts.most_common(1)[0][0]


def create_group_aware_splits(
    records: list[ImageRecord],
    config: DatasetConfig,
    *,
    force_recompute: bool = False,
) -> SplitResult:
    preserve = bool(deep_get(config, "splits", "preserve_official_splits", default=True))
    ratios = config.split_ratios
    seed = config.split_seed
    warnings: list[str] = []

    usable = [r for r in records if r.readable and r.label is not None and "missing_file" not in r.problems]
    official_audit = audit_official_splits(usable)

    assignments: dict[str, str] = {}
    mode = "group_stratified_70_15_15"

    if preserve and official_audit.get("present") and official_audit.get("groups_spanning_splits", 1) == 0:
        mode = "official_preserved"
        for record in usable:
            if record.split_official:
                assignments[record.image_path] = record.split_official
        # Any usable image without official split falls through to remaining pool.
        remaining = [r for r in usable if r.image_path not in assignments]
        if remaining:
            warnings.append(
                f"{len(remaining)} usable images lacked official split labels and were assigned via group split."
            )
            sub = _assign_group_stratified(remaining, ratios, seed)
            assignments.update(sub)
    elif preserve and official_audit.get("present") and official_audit.get("groups_spanning_splits", 0) > 0:
        warnings.append(
            "Official splits exist but groups span multiple splits "
            f"({official_audit['groups_spanning_splits']} groups). "
            "Falling back to reproducible group-aware 70/15/15 to prevent leakage."
        )
        assignments = _assign_group_stratified(usable, ratios, seed)
    else:
        assignments = _assign_group_stratified(usable, ratios, seed)

    # Metadata quality note
    has_specimen = any(r.specimen_id for r in usable)
    has_session = any(r.capture_session for r in usable)
    if not has_specimen and not has_session:
        warnings.append(str(deep_get(config, "grouping", "residual_leakage_warning", default="")).strip())

    split_counts = Counter(assignments.values())
    label_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    path_to_label = {r.image_path: r.label for r in usable}
    for path, split in assignments.items():
        label = path_to_label.get(path)
        if label:
            label_by_split[split][label] += 1

    fingerprint_payload = {
        "mode": mode,
        "seed": seed,
        "ratios": ratios,
        "assignments": dict(sorted(assignments.items())),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    summary = {
        "mode": mode,
        "seed": seed,
        "ratios": ratios,
        "split_counts": dict(split_counts),
        "label_counts_by_split": {k: dict(v) for k, v in sorted(label_by_split.items())},
        "official_split_audit": official_audit,
        "usable_images": len(usable),
        "assigned_images": len(assignments),
    }
    return SplitResult(
        assignments=assignments,
        mode=mode,
        fingerprint=fingerprint,
        summary=summary,
        leakage_warnings=[w for w in warnings if w],
    )


def _assign_group_stratified(
    records: list[ImageRecord],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        groups[_group_key(record)].append(record)

    by_label_groups: dict[str, list[str]] = defaultdict(list)
    for group_id, items in groups.items():
        by_label_groups[_majority_label(items)].append(group_id)

    rng = random.Random(seed)
    assignments: dict[str, str] = {}
    for label, group_ids in by_label_groups.items():
        ordered = list(group_ids)
        rng.shuffle(ordered)
        n = len(ordered)
        n_train = int(n * ratios.get("train", 0.7))
        n_val = int(n * ratios.get("val", 0.15))
        # Remainder to test to ensure all groups assigned.
        train_ids = set(ordered[:n_train])
        val_ids = set(ordered[n_train : n_train + n_val])
        test_ids = set(ordered[n_train + n_val :])
        for group_id in ordered:
            if group_id in train_ids:
                split = "train"
            elif group_id in val_ids:
                split = "val"
            else:
                split = "test"
            for record in groups[group_id]:
                assignments[record.image_path] = split
    return assignments

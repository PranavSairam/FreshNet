from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .integrity import ManifestRow, load_manifest_rows, stable_split_group_key, write_class_mapping
from .prepare import MANIFEST_FIELDS


def regenerate_group_aware_splits(
    manifests_dir: Path,
    *,
    ratios: dict[str, float] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Rebuild train/val/test using exact-hash + heuristic groups (near-dup excluded).

    Preserves existing rows/metadata; only updates `split` and the fingerprint.
    """
    ratios = ratios or {"train": 0.70, "val": 0.15, "test": 0.15}
    rows = load_manifest_rows(manifests_dir)
    if not rows:
        raise RuntimeError("No manifest rows to regenerate.")

    # Load full CSV dicts to preserve all columns.
    all_path = manifests_dir / "all.csv"
    with all_path.open(newline="", encoding="utf-8") as handle:
        dict_rows = list(csv.DictReader(handle))

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for raw, typed in zip(dict_rows, rows):
        key = stable_split_group_key(typed)
        groups[key].append(raw)

    by_label_groups: dict[str, list[str]] = defaultdict(list)
    for key, members in groups.items():
        labels = Counter(m["label"] for m in members)
        majority = labels.most_common(1)[0][0]
        by_label_groups[majority].append(key)

    rng = random.Random(seed)
    assignments: dict[str, str] = {}  # image_path -> split
    for _label, group_ids in by_label_groups.items():
        ordered = list(group_ids)
        rng.shuffle(ordered)
        n = len(ordered)
        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["val"])
        for idx, group_id in enumerate(ordered):
            if idx < n_train:
                split = "train"
            elif idx < n_train + n_val:
                split = "val"
            else:
                split = "test"
            for member in groups[group_id]:
                assignments[member["image_path"]] = split
                member["split"] = split
                # Normalize group_id stored in manifest to the leakage key.
                member["group_id"] = group_id

    fingerprint_payload = {
        "mode": "group_stratified_70_15_15_exact_and_heuristic",
        "seed": seed,
        "ratios": ratios,
        "near_affects_splits": False,
        "assignments": dict(sorted(assignments.items())),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    def write_split(name: str, items: list[dict[str, str]]) -> None:
        path = manifests_dir / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for row in items:
                writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})

    write_split("all", dict_rows)
    write_split("train", [r for r in dict_rows if r["split"] == "train"])
    write_split("val", [r for r in dict_rows if r["split"] == "val"])
    write_split("test", [r for r in dict_rows if r["split"] == "test"])

    mapping_path = write_class_mapping(manifests_dir)
    fingerprint_path = manifests_dir / "split_fingerprint.json"
    split_counts = Counter(r["split"] for r in dict_rows)
    fingerprint_path.write_text(
        json.dumps(
            {
                "split_fingerprint": fingerprint,
                "canonical_labels": ["Fresh", "Spoiled"],
                "class_mapping": {"Fresh": 0, "Spoiled": 1},
                "num_classes": 2,
                "three_class_ready": False,
                "near_affects_splits": False,
                "mode": fingerprint_payload["mode"],
                "seed": seed,
                "ratios": ratios,
                "split_counts": dict(split_counts),
                "class_mapping_path": str(mapping_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "fingerprint": fingerprint,
        "split_counts": dict(split_counts),
        "n_groups": len(groups),
        "fingerprint_path": str(fingerprint_path),
    }

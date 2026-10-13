#!/usr/bin/env python3
"""Build a versioned source-image split without touching the active manifests.

Confirmed links require decoded-pixel identity or a structural match after the
alignment implied by a checked filename rule or a flip. Average-hash equality
and low RGB error are not enough.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_MANIFESTS = ROOT / "data" / "processed" / "manifests"
OUT = ROOT / "data" / "processed" / "manifests_v3_source"
CANDIDATES = ROOT / "data" / "processed" / "reports" / "near_review" / "all_candidates.jsonl"
HASH_CACHE = ROOT / "data" / "processed" / "cache" / "image_hashes.jsonl"
ACTIVE_FP = "005d750025b30ce1074333714f7e2ee7b89cc0ec9acd4ba56f62010edd5850f9"

PREFIXES = (
    "rotated_by_15_",
    "rotated_by_30_",
    "rotated_by_45_",
    "rotated_by_60_",
    "rotated_by_75_",
    "vertical_flip_",
    "horizontal_flip_",
    "translation_",
    "saltandpepper_",
    "salt_and_pepper_",
)
CONFIRM_NCC = 0.992
DISTINCT_NCC = 0.90
FIELDS = [
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


def strip_prefixes(name: str) -> tuple[str, list[str]]:
    applied: list[str] = []
    changed = True
    while changed:
        changed = False
        for prefix in PREFIXES:
            if name.startswith(prefix):
                applied.append(prefix[:-1])
                name = name[len(prefix) :]
                changed = True
    return name, applied


def feature(path: str, cache: dict[str, np.ndarray]) -> np.ndarray:
    found = cache.get(path)
    if found is not None:
        return found
    with Image.open(path) as image:
        array = np.asarray(image.convert("L").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32)
    array = array - float(array.mean())
    norm = float(np.linalg.norm(array))
    vector = array / norm if norm else array
    cache[path] = vector
    return vector


def ncc(path_a: str, path_b: str, cache: dict[str, np.ndarray]) -> float:
    left = feature(path_a, cache)
    right = feature(path_b, cache)
    return float(np.dot(left.ravel(), right.ravel()))


def feature_image(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("L").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32)
    array = array - float(array.mean())
    norm = float(np.linalg.norm(array))
    return array / norm if norm else array


def ncc_images(left: Image.Image, right: Image.Image) -> float:
    return float(np.dot(feature_image(left).ravel(), feature_image(right).ravel()))


def aligned_ncc(aug_path: str, base_path: str, prefixes: list[str], cache: dict[str, np.ndarray]) -> tuple[float, str]:
    """Score the filename rule by undoing the claimed transform, plus flips."""
    del cache
    with Image.open(aug_path) as image, Image.open(base_path) as base_image:
        rgb = image.convert("RGB")
        base = base_image.convert("RGB")
    variants: list[tuple[str, Image.Image]] = [("identity", rgb)]
    variants.append(("horizontal_flip", rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)))
    variants.append(("vertical_flip", rgb.transpose(Image.Transpose.FLIP_TOP_BOTTOM)))
    for prefix in prefixes:
        if prefix.startswith("rotated_by_"):
            angle = int(prefix.split("_")[2])
            variants.append((f"rotate_{angle}", rgb.rotate(-angle, expand=False, fillcolor=(255, 255, 255))))
    best_score = -1.0
    best_name = "identity"
    for name, variant in variants:
        score = ncc_images(variant, base)
        if score > best_score:
            best_score = score
            best_name = name
    if "translation" in prefixes:
        with Image.open(base_path) as base_image:
            base = np.asarray(base_image.convert("L"), dtype=np.float32)
        aug = np.asarray(rgb.convert("L"), dtype=np.float32)
        if aug.shape == base.shape:
            best_shift = best_score
            height, width = aug.shape
            for dy in range(-36, 37, 4):
                for dx in range(-36, 37, 4):
                    shifted = np.roll(np.roll(aug, dy, 0), dx, 1)
                    ys = slice(36, height - 36)
                    xs = slice(36, width - 36)
                    if ys.start >= ys.stop or xs.start >= xs.stop:
                        continue
                    left = shifted[ys, xs]
                    right = base[ys, xs]
                    left = left - left.mean()
                    right = right - right.mean()
                    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
                    score = float((left * right).sum() / denom) if denom else 0.0
                    if score > best_shift:
                        best_shift = score
                        best_name = f"shift_{dy}_{dx}"
            best_score = best_shift
    return best_score, best_name


def pixels_equal(path_a: str, path_b: str) -> bool:
    with Image.open(path_a) as left, Image.open(path_b) as right:
        left_rgb = np.asarray(left.convert("RGB"))
        right_rgb = np.asarray(right.convert("RGB"))
    return left_rgb.shape == right_rgb.shape and bool(np.array_equal(left_rgb, right_rgb))


def main() -> int:
    rows = list(csv.DictReader(ACTIVE_MANIFESTS.joinpath("all.csv").open(encoding="utf-8")))
    if not rows:
        raise SystemExit("active manifest is empty")
    phash: dict[str, int] = {}
    with HASH_CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("phash"):
                phash[item["image_path"]] = int(item["phash"], 16)

    by_name: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_food: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_path = {row["image_path"]: row for row in rows}
    for row in rows:
        by_name[(row["food_type"], Path(row["image_path"]).name)].append(row)
        by_food[row["food_type"]].append(row)

    parent = {row["image_path"]: row["image_path"] for row in rows}

    def find(path: str) -> str:
        while parent[path] != path:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    sha_groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        sha_groups[row["content_sha256"]].append(row["image_path"])
    for members in sha_groups.values():
        for other in members[1:]:
            union(members[0], other)

    evidence_path = OUT / "grouping_evidence.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    decisions: list[dict[str, object]] = []
    cache: dict[str, np.ndarray] = {}
    seen_pairs: set[tuple[str, str]] = set()

    def consider(path_a: str, path_b: str, proposal: str, prefixes: list[str] | None = None) -> None:
        if path_a == path_b:
            return
        key = tuple(sorted((path_a, path_b)))
        if key in seen_pairs:
            return
        seen_pairs.add(key)
        if prefixes:
            score, alignment = aligned_ncc(path_a, path_b, prefixes, cache)
        else:
            direct = ncc(path_a, path_b, cache)
            with Image.open(path_a) as image, Image.open(path_b) as other:
                flipped_score = ncc_images(image.convert("RGB").transpose(Image.Transpose.FLIP_LEFT_RIGHT), other.convert("RGB"))
            if flipped_score > direct:
                score, alignment = flipped_score, "horizontal_flip"
            else:
                score, alignment = direct, "identity"
        translation_ok = bool(prefixes and "translation" in prefixes and alignment.startswith("shift_") and score >= 0.95)
        if score >= 0.999 and pixels_equal(path_a, path_b):
            decision = "confirmed_same_capture"
            reason = "decoded RGB pixels are identical"
            score = 1.0
            alignment = "identity"
            union(path_a, path_b)
        elif score >= CONFIRM_NCC or translation_ok:
            decision = "confirmed_same_capture"
            reason = f"structural match {score:.4f} after {alignment}; hash or RGB error alone was not the criterion"
            union(path_a, path_b)
        elif score < DISTINCT_NCC:
            decision = "distinct"
            reason = f"structural score {score:.4f} below {DISTINCT_NCC}; not merged"
        else:
            decision = "unresolved"
            reason = f"structural score {score:.4f} is between distinct and confirmed"
        quarantine: list[str] = []
        if decision != "confirmed_same_capture" and prefixes:
            quarantine = [path_a]
            reason = f"{reason} Augmented file quarantined; canonical file kept unless another unresolved link removes it."
        elif decision == "unresolved":
            quarantine = [path_a, path_b]
        decisions.append(
            {
                "image_a": path_a,
                "image_b": path_b,
                "proposal": proposal,
                "decision": decision,
                "score": round(score, 4),
                "alignment": alignment,
                "reason": reason,
                "label_a": by_path[path_a]["label"],
                "label_b": by_path[path_b]["label"],
                "quarantine": quarantine,
            }
        )

    filename_pairs = 0
    for row in rows:
        name = Path(row["image_path"]).name
        stripped, applied = strip_prefixes(name)
        if not applied:
            continue
        mates = by_name.get((row["food_type"], stripped), [])
        for mate in mates:
            if mate["image_path"] == row["image_path"]:
                continue
            filename_pairs += 1
            consider(row["image_path"], mate["image_path"], "filename_convention", applied)

    alias_checked = 0
    for food, members in by_food.items():
        hashes = [(member["image_path"], phash.get(member["image_path"])) for member in members]
        hashes = [(path, value) for path, value in hashes if value is not None]
        for member in members:
            name = Path(member["image_path"]).name
            alias = name.startswith(("a_f", "b_f")) or ("(" in Path(name).stem and Path(name).stem.endswith(")"))
            if not alias:
                continue
            origin = phash.get(member["image_path"])
            if origin is None:
                continue
            ranked = sorted(
                ((int((origin ^ other).bit_count()), other_path) for other_path, other in hashes if other_path != member["image_path"]),
                key=lambda item: item[0],
            )[:3]
            for distance, other_path in ranked:
                if distance > 4:
                    break
                alias_checked += 1
                consider(member["image_path"], other_path, f"renamed_candidate_hamming_{distance}", None)

    hamming0_cross_group = 0
    with CANDIDATES.open(encoding="utf-8") as handle:
        for line in handle:
            pair = json.loads(line)
            if int(pair["hamming"]) != 0:
                continue
            left, right = pair["image_a"], pair["image_b"]
            if left not in parent or right not in parent:
                continue
            if find(left) == find(right):
                continue
            hamming0_cross_group += 1
            consider(left, right, "existing_hamming0_candidate", None)

    counts = Counter(str(item["decision"]) for item in decisions)
    with evidence_path.open("w", encoding="utf-8") as handle:
        for item in decisions:
            handle.write(json.dumps(item) + "\n")

    unresolved_images = {path for item in decisions for path in item["quarantine"]}  # type: ignore[index]
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["image_path"] in unresolved_images:
            continue
        groups[find(row["image_path"])].append(row)

    conflict_images: set[str] = set()
    conflict_groups = 0
    for members in groups.values():
        labels = {member["label"] for member in members}
        if len(labels) > 1:
            conflict_groups += 1
            conflict_images.update(member["image_path"] for member in members)
    for path in conflict_images:
        root = find(path)
        groups.pop(root, None)

    exclusion_reasons: dict[str, str] = {}
    for item in decisions:
        for path in item["quarantine"]:  # type: ignore[index]
            exclusion_reasons.setdefault(str(path), str(item["reason"]))
    excluded = []
    for path in sorted(unresolved_images):
        excluded.append({"image_path": path, "reason": exclusion_reasons.get(path, "unresolved candidate")})
    for path in sorted(conflict_images):
        excluded.append({"image_path": path, "reason": "source group contains conflicting Fresh/Spoiled labels"})

    by_label: dict[str, list[str]] = defaultdict(list)
    for root, members in groups.items():
        label = Counter(member["label"] for member in members).most_common(1)[0][0]
        by_label[label].append(root)
    rng = random.Random(42)
    assignments: dict[str, str] = {}
    for _label, roots in by_label.items():
        ordered = list(roots)
        rng.shuffle(ordered)
        n_train = int(len(ordered) * 0.70)
        n_val = int(len(ordered) * 0.15)
        for index, root in enumerate(ordered):
            split = "train" if index < n_train else "val" if index < n_train + n_val else "test"
            for member in groups[root]:
                member["split"] = split
                member["group_id"] = f"source:{root}"
                assignments[member["image_path"]] = split

    kept = [row for row in rows if row["image_path"] in assignments]
    for name, items in (
        ("all", kept),
        ("train", [row for row in kept if row["split"] == "train"]),
        ("val", [row for row in kept if row["split"] == "val"]),
        ("test", [row for row in kept if row["split"] == "test"]),
        ("excluded", []),
    ):
        pass
    def write_rows(path: Path, items: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(items)

    write_rows(OUT / "all.csv", kept)
    for split in ("train", "val", "test"):
        write_rows(OUT / f"{split}.csv", [row for row in kept if row["split"] == split])
    with (OUT / "exclusions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "reason"])
        writer.writeheader()
        writer.writerows(excluded)

    confirmed_cross = 0
    confirmed_checked = 0
    for item in decisions:
        if item["decision"] != "confirmed_same_capture":
            continue
        confirmed_checked += 1
        left = assignments.get(str(item["image_a"]))
        right = assignments.get(str(item["image_b"]))
        if left and right and left != right:
            confirmed_cross += 1

    payload = {
        "mode": "v3_verified_source_groups",
        "seed": 42,
        "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "confirm_ncc": CONFIRM_NCC,
        "translation_confirm_ncc": 0.95,
        "distinct_ncc": DISTINCT_NCC,
        "assignments": dict(sorted(assignments.items())),
        "excluded": sorted(path for path in unresolved_images | conflict_images),
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    image_counts = Counter(row["split"] for row in kept)
    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for root, members in groups.items():
        if not members or members[0]["image_path"] not in assignments:
            continue
        split = members[0]["split"]
        label = Counter(member["label"] for member in members).most_common(1)[0][0]
        group_counts[split][label] += 1
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in kept:
        class_counts[row["split"]][row["label"]] += 1
    sizes = sorted((len(members) for members in groups.values()), reverse=True)
    report = {
        "split_fingerprint": fingerprint,
        "active_training_fingerprint_unchanged": ACTIVE_FP,
        "active_manifests_modified": False,
        "image_counts": {split: dict(class_counts[split]) for split in ("train", "val", "test")},
        "image_totals": dict(image_counts),
        "group_counts": {split: dict(group_counts[split]) for split in ("train", "val", "test")},
        "group_total": len(groups),
        "largest_groups": sizes[:15],
        "decision_counts": dict(counts),
        "filename_convention_pairs_seen": filename_pairs,
        "alias_candidate_checks": alias_checked,
        "hamming0_pairs_across_filename_groups": hamming0_cross_group,
        "confirmed_relationships_checked": confirmed_checked,
        "confirmed_relationships_crossing_splits": confirmed_cross,
        "excluded_images": len(excluded),
        "label_conflict_groups": conflict_groups,
        "original_images": len(rows),
        "benchmark_images": len(kept),
        "coverage_limits": [
            "The saved candidate search compares at most 64 images inside each average-hash band. It is not exhaustive.",
            "Renamed-copy coverage adds the three nearest same-food hash neighbors within Hamming 4 for a_f/b_f and parenthetical names.",
            "Other renamed copies are verified only when they already appear as Hamming-0 candidates.",
            "Passing the known confirmed links does not mean every same-capture pair was found.",
        ],
    }
    (OUT / "split_fingerprint.json").write_text(json.dumps({"split_fingerprint": fingerprint, **{k: report[k] for k in report if k != "split_fingerprint"}}, indent=2), encoding="utf-8")
    (OUT / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

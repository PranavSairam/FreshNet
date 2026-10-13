#!/usr/bin/env python3
"""Review cross-split perceptual candidates without changing manifests or splits."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / "ml"
if str(ML) not in sys.path:
    sys.path.insert(0, str(ML))

from dataset.duplicates import hamming
from dataset.integrity import (  # noqa: E402
    load_hash_cache,
    load_manifest_rows,
    scalable_near_candidates,
    write_near_scan_provenance,
)
from dataset.config import load_config  # noqa: E402

REP_CSV = ROOT / "data/processed/reports/near_duplicate_candidates.csv"
OUT_DIR = ROOT / "data/processed/reports/near_review"
FINGERPRINT = "005d750025b30ce1074333714f7e2ee7b89cc0ec9acd4ba56f62010edd5850f9"
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


def load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def strip_name(path: str) -> tuple[str, str, list[str]]:
    file_path = Path(path)
    name = file_path.name
    applied: list[str] = []
    changed = True
    while changed:
        changed = False
        for prefix in PREFIXES:
            if name.startswith(prefix):
                applied.append(prefix[:-1])
                name = name[len(prefix) :]
                changed = True
    return file_path.parent.name, Path(name).stem, applied


def exact_after_resize(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape == right.shape:
        return False
    target = (right.shape[1], right.shape[0])
    resized = np.asarray(Image.fromarray(left).resize(target, Image.Resampling.NEAREST))
    if resized.shape == right.shape and np.array_equal(resized, right):
        return True
    target = (left.shape[1], left.shape[0])
    resized = np.asarray(Image.fromarray(right).resize(target, Image.Resampling.NEAREST))
    return resized.shape == left.shape and np.array_equal(resized, left)


def flip_equal(left: np.ndarray, right: np.ndarray) -> str | None:
    if left.shape != right.shape:
        return None
    if np.array_equal(np.fliplr(left), right):
        return "horizontal_flip"
    if np.array_equal(np.flipud(left), right):
        return "vertical_flip"
    return None


def classify_pair(row: dict[str, str]) -> dict[str, object]:
    left_path = row["image_a"]
    right_path = row["image_b"]
    left = load_rgb(left_path)
    right = load_rgb(right_path)
    left_sha = row.get("sha_a")
    right_sha = row.get("sha_b")
    food_a, stem_a, aug_a = strip_name(left_path)
    food_b, stem_b, aug_b = strip_name(right_path)
    pixels_equal = left.shape == right.shape and bool(np.array_equal(left, right))
    resized = exact_after_resize(left, right)
    flipped = flip_equal(left, right)
    named_same = food_a == food_b and stem_a == stem_b
    if left_sha and right_sha and left_sha == right_sha:
        verdict = "confirmed_duplicate"
        evidence = "identical SHA-256 file bytes"
    elif pixels_equal:
        verdict = "confirmed_duplicate"
        evidence = "decoded RGB pixels are equal; file bytes or metadata differ"
    elif resized:
        verdict = "confirmed_duplicate"
        evidence = "decoded RGB matches exactly after nearest-neighbor resize"
    elif flipped:
        verdict = "likely_related"
        evidence = f"decoded RGB matches a {flipped}"
    elif named_same and (aug_a or aug_b):
        verdict = "likely_related"
        evidence = (
            "filename augmentation prefix strips to the same food folder and stem; "
            "pixels are not identical"
        )
    elif named_same:
        verdict = "likely_related"
        evidence = "same food folder and filename stem; pixels are not identical"
    else:
        verdict = "unresolved"
        evidence = (
            "Hamming distance alone is not duplication. No pixel, resize, flip, "
            "or filename-stem identity was found."
        )
    return {
        "image_a": left_path,
        "image_b": right_path,
        "label_a": row["label_a"],
        "label_b": row["label_b"],
        "split_a": row["split_a"],
        "split_b": row["split_b"],
        "hamming": int(row["hamming"]),
        "size_a": [int(left.shape[1]), int(left.shape[0])],
        "size_b": [int(right.shape[1]), int(right.shape[0])],
        "pixels_equal": pixels_equal,
        "resized_exact": resized,
        "flip_match": flipped,
        "food_a": food_a,
        "food_b": food_b,
        "stem_a": stem_a,
        "stem_b": stem_b,
        "aug_a": aug_a,
        "aug_b": aug_b,
        "verdict": verdict,
        "evidence": evidence,
    }


def contact_sheets(pairs: list[dict[str, object]], out_dir: Path, per_sheet: int = 4) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    thumb = 180
    row_h = 210
    width = 980
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for sheet_index in range(0, len(pairs), per_sheet):
        chunk = pairs[sheet_index : sheet_index + per_sheet]
        canvas = Image.new("RGB", (width, row_h * len(chunk) + 8), "white")
        draw = ImageDraw.Draw(canvas)
        for row_index, pair in enumerate(chunk):
            y = 4 + row_index * row_h
            for column, key in enumerate(("image_a", "image_b")):
                with Image.open(str(pair[key])) as image:
                    image = image.convert("RGB")
                    image.thumbnail((thumb, thumb))
                    canvas.paste(image, (8 + column * (thumb + 8), y))
            text = (
                f"{pair['verdict']}  hamming={pair['hamming']}  "
                f"{pair['split_a']}/{pair['label_a']} vs {pair['split_b']}/{pair['label_b']}\n"
                f"A {pair['size_a']} {Path(str(pair['image_a'])).name}\n"
                f"B {pair['size_b']} {Path(str(pair['image_b'])).name}\n"
                f"{pair['evidence']}"
            )
            draw.multiline_text((8 + 2 * (thumb + 8), y + 4), text, fill="black", font=font, spacing=3)
        path = out_dir / f"contact_sheet_{sheet_index // per_sheet + 1:02d}.png"
        canvas.save(path)
        paths.append(str(path))
    return paths


def review_representative() -> dict[str, object]:
    rows = list(csv.DictReader(REP_CSV.open(encoding="utf-8")))
    cross = [row for row in rows if row["cross_split"] == "True"]
    reviewed = [classify_pair(row) for row in cross]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "representative_cross_split_review.json").write_text(
        json.dumps(reviewed, indent=2), encoding="utf-8"
    )
    sheets = contact_sheets(reviewed, OUT_DIR / "sheets")
    counts = Counter(str(item["verdict"]) for item in reviewed)
    summary = {
        "representative_csv_rows": len(rows),
        "cross_split_rows_reviewed": len(reviewed),
        "verdict_counts": dict(counts),
        "contact_sheets": sheets,
        "note": (
            "These  rows are the representative CSV only, not the full candidate search. "
            "Hamming 0 was not treated as proof of duplication."
        ),
    }
    (OUT_DIR / "representative_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def recover_full_candidates() -> None:
    config = load_config(ROOT / "configs" / "dataset.yaml")
    rows = load_manifest_rows(config.manifests_dir)
    cache = load_hash_cache(ROOT / "data/processed/cache/image_hashes.jsonl")
    all_path = OUT_DIR / "all_candidates.jsonl"
    cross_path = OUT_DIR / "cross_split_candidates.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = scalable_near_candidates(
        rows,
        cache,
        band_chars=3,
        threshold=5,
        max_candidates_per_image=64,
        representative_limit=200,
        all_pairs_jsonl=all_path,
        cross_split_jsonl=cross_path,
    )
    groups = connected_groups(cross_path)
    payload = {
        "dataset_fingerprint": FINGERPRINT,
        "hash_cache_path": str(ROOT / "data/processed/cache/image_hashes.jsonl"),
        "images_with_phash": summary.get("images_with_phash"),
        "candidate_pairs": summary.get("candidate_pairs"),
        "cross_split_candidate_pairs": summary.get("cross_split_candidate_pairs"),
        "candidate_search_complete": bool(summary.get("complete")),
        "hash_computation_complete": summary.get("images_with_phash") == len(rows),
        "hamming_histogram": summary.get("hamming_histogram"),
        "cross_split_hamming_histogram": summary.get("cross_split_hamming_histogram"),
        "scan_configuration": {
            "method": "banded_ahash_lsh",
            "hash": "average hash",
            "hash_size": 8,
            "band_prefix_hex_chars": 3,
            "hamming_threshold": 5,
            "max_candidates_per_image": 64,
            "note": (
                "Recovered with the same banded, capped search as the original 418,572-pair run. "
                "Not all image pairs. Hamming distance is not specimen identity."
            ),
        },
        "all_candidates_jsonl": str(all_path),
        "cross_split_candidates_jsonl": str(cross_path),
        "cross_split_connected_groups": groups,
        "review_status_path": str(ROOT / "data/processed/reports/near_duplicate_review_status.json"),
    }
    write_near_scan_provenance(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_fingerprint": FINGERPRINT,
            "hash_cache_path": payload["hash_cache_path"],
            "images_with_phash": payload["images_with_phash"],
            "candidate_pairs": payload["candidate_pairs"],
            "cross_split_candidate_pairs": payload["cross_split_candidate_pairs"],
            "candidate_search_complete": payload["candidate_search_complete"],
            "hash_computation_complete": payload["hash_computation_complete"],
            "scan_configuration": payload["scan_configuration"],
            "all_candidates_jsonl": str(all_path),
            "cross_split_candidates_jsonl": str(cross_path),
            "recovered_from": "cached average hashes; manifests were not modified",
            "review_status_path": payload["review_status_path"],
        }
    )
    (OUT_DIR / "full_candidate_summary.json").write_text(json.dumps({**payload, **{"connected_group_preview": groups}}, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("candidate_pairs", "cross_split_candidate_pairs", "hamming_histogram", "cross_split_hamming_histogram", "candidate_search_complete")}, indent=2))


def connected_groups(path: Path) -> dict[str, object]:
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    distance_edges = Counter()
    closest: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            pair = json.loads(line)
            distance_edges[int(pair["hamming"])] += 1
            if int(pair["hamming"]) == 0:
                union(pair["image_a"], pair["image_b"])
                if len(closest) < 30:
                    closest.append(
                        {
                            "hamming": 0,
                            "split_a": pair["split_a"],
                            "split_b": pair["split_b"],
                            "image_a": Path(pair["image_a"]).name,
                            "image_b": Path(pair["image_b"]).name,
                        }
                    )
    members: dict[str, list[str]] = defaultdict(list)
    for item in list(parent):
        members[find(item)].append(item)
    multi = [sorted(group) for group in members.values() if len(group) > 1]
    multi.sort(key=len, reverse=True)
    return {
        "edge_rule": "cross-split pairs with average-hash Hamming distance 0 only",
        "not_applied_to_splits": True,
        "warning": "These groups are perceptual candidates. They were not merged into manifests.",
        "groups": len(multi),
        "largest_group_sizes": [len(group) for group in multi[:20]],
        "closest_cross_split_examples": closest,
    }


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "representative"
    if mode == "representative":
        review_representative()
        return 0
    if mode == "full":
        recover_full_candidates()
        return 0
    raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    raise SystemExit(main())

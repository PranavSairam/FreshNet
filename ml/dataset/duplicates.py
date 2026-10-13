from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

from PIL import Image

from .config import DatasetConfig, deep_get
from .records import ImageRecord


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(path: Path, hash_size: int = 8) -> str:
    with Image.open(path) as img:
        gray = img.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        pixels = list(getattr(gray, "get_flattened_data", gray.getdata)())
    mean = sum(pixels) / len(pixels)
    bits = "".join("1" if pixel >= mean else "0" for pixel in pixels)
    # Pack to hex
    value = int(bits, 2)
    width = (hash_size * hash_size + 3) // 4
    return f"{value:0{width}x}"


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b)) * 4
    x = int(a, 16) ^ int(b, 16)
    return x.bit_count()


def detect_exact_duplicates(records: list[ImageRecord]) -> dict:
    by_hash: dict[str, list[ImageRecord]] = defaultdict(list)
    for index, record in enumerate(records, start=1):
        if index == 1 or index % 2000 == 0 or index == len(records):
            print(f"[exact-hash] {index}/{len(records)}", flush=True)
        path = Path(record.image_path)
        if not path.is_file() or record.readable is False:
            continue
        try:
            digest = file_sha256(path)
        except OSError:
            record.problems.append("hash_failed")
            continue
        record.content_sha256 = digest
        by_hash[digest].append(record)

    duplicate_groups = 0
    duplicate_images = 0
    for digest, group in by_hash.items():
        if len(group) < 2:
            continue
        duplicate_groups += 1
        duplicate_images += len(group) - 1
        primary = group[0].image_path
        for extra in group[1:]:
            extra.exact_duplicate_of = primary
            if "exact_duplicate" not in extra.problems:
                extra.problems.append("exact_duplicate")
            # Merge groups for leakage prevention.
            for item in group:
                item.group_id = f"exact:{digest}"

    return {
        "exact_duplicate_groups": duplicate_groups,
        "exact_duplicate_extra_images": duplicate_images,
    }


def detect_near_duplicates(records: list[ImageRecord], config: DatasetConfig) -> dict:
    enabled = bool(deep_get(config, "duplicates", "enable_near", default=True))
    if not enabled:
        return {"near_duplicate_groups": 0, "near_duplicate_pairs_flagged": 0, "skipped": True}

    hash_size = int(deep_get(config, "duplicates", "near_hash_size", default=8))
    threshold = int(deep_get(config, "duplicates", "near_hamming_threshold", default=4))
    max_images = int(deep_get(config, "duplicates", "near_max_images", default=20000))

    candidates = [r for r in records if r.readable and Path(r.image_path).is_file()]
    truncated = False
    if len(candidates) > max_images:
        candidates = candidates[:max_images]
        truncated = True

    hashed: list[tuple[ImageRecord, str]] = []
    for record in candidates:
        try:
            phash = average_hash(Path(record.image_path), hash_size=hash_size)
            record.phash = phash
            hashed.append((record, phash))
        except OSError:
            record.problems.append("phash_failed")

    # Bucket by high bits to avoid O(n^2) over the full set.
    buckets: dict[str, list[tuple[ImageRecord, str]]] = defaultdict(list)
    for record, phash in hashed:
        buckets[phash[:3]].append((record, phash))

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pairs = 0
    for bucket in buckets.values():
        for i in range(len(bucket)):
            left_rec, left_hash = bucket[i]
            for j in range(i + 1, len(bucket)):
                right_rec, right_hash = bucket[j]
                if hamming(left_hash, right_hash) <= threshold:
                    pairs += 1
                    union(left_rec.image_path, right_rec.image_path)

    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    path_to_record = {r.image_path: r for r, _ in hashed}
    for record, _ in hashed:
        root = find(record.image_path)
        groups[root].append(record)

    near_groups = 0
    for root, members in groups.items():
        if len(members) < 2:
            continue
        near_groups += 1
        group_name = f"near:{root}"
        for member in members:
            member.near_duplicate_group = group_name
            if "near_duplicate" not in member.problems:
                member.problems.append("near_duplicate")
            # Keep specimen grouping connected to near-duplicate clusters.
            member.group_id = f"{member.group_id}|{group_name}"

    return {
        "near_duplicate_groups": near_groups,
        "near_duplicate_pairs_flagged": pairs,
        "near_scan_images": len(hashed),
        "near_scan_truncated": truncated,
        "skipped": False,
    }


def run_duplicate_detection(
    records: list[ImageRecord],
    config: DatasetConfig,
    *,
    enable_near: bool | None = None,
) -> dict:
    summary: dict = {}
    if deep_get(config, "duplicates", "enable_exact", default=True):
        summary["exact"] = detect_exact_duplicates(records)
    else:
        summary["exact"] = {"exact_duplicate_groups": 0, "exact_duplicate_extra_images": 0, "skipped": True}

    if enable_near is False:
        summary["near"] = {
            "near_duplicate_groups": 0,
            "near_duplicate_pairs_flagged": 0,
            "skipped": True,
        }
    else:
        summary["near"] = detect_near_duplicates(records, config)
    return summary

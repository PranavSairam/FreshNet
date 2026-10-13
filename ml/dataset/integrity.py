from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import REPO_ROOT, DatasetConfig, deep_get, load_config
from .duplicates import average_hash, file_sha256, hamming

CLASS_ORDER = ("Fresh", "Spoiled")
NEAR_SCAN_PROVENANCE = REPO_ROOT / "data" / "processed" / "reports" / "near_scan_provenance.json"
NEAR_REVIEW_STATUS = REPO_ROOT / "data" / "processed" / "reports" / "near_duplicate_review_status.json"


@dataclass
class ManifestRow:
    image_path: str
    label: str
    split: str
    group_id: str
    content_sha256: str
    near_duplicate_group: str
    food_type: str
    specimen_id: str = ""
    capture_session: str = ""


def load_manifest_rows(manifests_dir: Path) -> list[ManifestRow]:
    path = manifests_dir / "all.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    rows: list[ManifestRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                ManifestRow(
                    image_path=raw["image_path"],
                    label=raw["label"],
                    split=raw["split"],
                    group_id=raw.get("group_id") or raw["image_path"],
                    content_sha256=raw.get("content_sha256") or "",
                    near_duplicate_group=raw.get("near_duplicate_group") or "",
                    food_type=raw.get("food_type") or "",
                    specimen_id=raw.get("specimen_id") or "",
                    capture_session=raw.get("capture_session") or "",
                )
            )
    return rows


def stable_split_group_key(row: ManifestRow) -> str:
    """Grouping key used for leakage prevention.

    Exact content hashes always bind together. Near-duplicate links are NOT used
    here — perceptual similarity is a review candidate only.
    """
    if row.content_sha256:
        return f"exact:{row.content_sha256}"
    # Prefer heuristic/specimen portion of group_id before any near suffix.
    base = row.group_id.split("|near:")[0]
    return base or row.image_path


def count_images_and_groups(rows: list[ManifestRow]) -> dict[str, Any]:
    by_split_label_images: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_split_label_groups: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    by_split_images: Counter[str] = Counter()
    by_split_groups: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        key = stable_split_group_key(row)
        by_split_images[row.split] += 1
        by_split_groups[row.split].add(key)
        by_split_label_images[row.split][row.label] += 1
        by_split_label_groups[row.split][row.label].add(key)

    return {
        "images_by_split": dict(by_split_images),
        "unique_groups_by_split": {k: len(v) for k, v in sorted(by_split_groups.items())},
        "images_by_split_and_class": {
            split: dict(sorted(labels.items())) for split, labels in sorted(by_split_label_images.items())
        },
        "unique_groups_by_split_and_class": {
            split: {label: len(groups) for label, groups in sorted(labels.items())}
            for split, labels in sorted(by_split_label_groups.items())
        },
    }


def find_cross_split_groups(rows: list[ManifestRow], key_fn) -> list[dict[str, Any]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    members: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        key = key_fn(row)
        mapping[key].add(row.split)
        members[key].append(row.image_path)
    leaked = []
    for key, splits in mapping.items():
        if len(splits) > 1:
            leaked.append(
                {
                    "group_key": key,
                    "splits": sorted(splits),
                    "example_paths": members[key][:5],
                    "n_images": len(members[key]),
                }
            )
    return leaked


def load_hash_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not cache_path.is_file():
        return cache
    with cache_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            cache[item["image_path"]] = item
    return cache


def append_hash_cache(cache_path: Path, items: Iterable[dict[str, Any]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, sort_keys=True) + "\n")


def ensure_hashes(
    rows: list[ManifestRow],
    cache_path: Path,
    *,
    hash_size: int,
    resume: bool = True,
) -> dict[str, dict[str, Any]]:
    cache = load_hash_cache(cache_path) if resume else {}
    pending: list[dict[str, Any]] = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        if row.image_path in cache and cache[row.image_path].get("phash"):
            continue
        path = Path(row.image_path)
        item: dict[str, Any] = {
            "image_path": row.image_path,
            "content_sha256": row.content_sha256 or None,
            "phash": None,
            "error": None,
        }
        try:
            if not item["content_sha256"] and path.is_file():
                item["content_sha256"] = file_sha256(path)
            item["phash"] = average_hash(path, hash_size=hash_size)
        except OSError as exc:
            item["error"] = str(exc)
        cache[row.image_path] = item
        pending.append(item)
        if len(pending) >= 200:
            append_hash_cache(cache_path, pending)
            pending.clear()
            print(f"[hash-cache] {index}/{total}", flush=True)
    if pending:
        append_hash_cache(cache_path, pending)
        print(f"[hash-cache] {total}/{total}", flush=True)
    return cache


def scalable_near_candidates(
    rows: list[ManifestRow],
    cache: dict[str, dict[str, Any]],
    *,
    band_chars: int,
    threshold: int,
    max_candidates_per_image: int,
    representative_limit: int,
    all_pairs_jsonl: Path | None = None,
    cross_split_jsonl: Path | None = None,
) -> dict[str, Any]:
    """Banded Hamming search over cached perceptual hashes (not all-pairs)."""
    entries: list[tuple[str, str, str]] = []  # path, phash, label
    label_by_path = {r.image_path: r.label for r in rows}
    split_by_path = {r.image_path: r.split for r in rows}
    for row in rows:
        item = cache.get(row.image_path)
        if not item or not item.get("phash"):
            continue
        entries.append((row.image_path, item["phash"], row.label))

    bands: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path, phash, _ in entries:
        bands[phash[:band_chars]].append((path, phash))

    pairs: list[dict[str, Any]] = []
    seen_pair: set[tuple[str, str]] = set()
    hamming_histogram: Counter[int] = Counter()
    cross_split_histogram: Counter[int] = Counter()
    all_handle = all_pairs_jsonl.open("w", encoding="utf-8") if all_pairs_jsonl else None
    cross_handle = cross_split_jsonl.open("w", encoding="utf-8") if cross_split_jsonl else None
    for band_items in bands.values():
        # Limit pairwise work inside each band.
        for i, (left_path, left_hash) in enumerate(band_items):
            compared = 0
            for right_path, right_hash in band_items[i + 1 :]:
                if compared >= max_candidates_per_image:
                    break
                compared += 1
                dist = hamming(left_hash, right_hash)
                if dist > threshold:
                    continue
                a, b = sorted((left_path, right_path))
                key = (a, b)
                if key in seen_pair:
                    continue
                seen_pair.add(key)
                cross_split = split_by_path.get(a) != split_by_path.get(b)
                pair = {
                    "image_a": a,
                    "image_b": b,
                    "hamming": dist,
                    "label_a": label_by_path.get(a),
                    "label_b": label_by_path.get(b),
                    "split_a": split_by_path.get(a),
                    "split_b": split_by_path.get(b),
                    "cross_split": cross_split,
                    "same_label": label_by_path.get(a) == label_by_path.get(b),
                    "note": (
                        "Perceptual candidate for human review only. "
                        "Not proof that specimens are identical."
                    ),
                }
                pairs.append(pair)
                hamming_histogram[dist] += 1
                if cross_split:
                    cross_split_histogram[dist] += 1
                if all_handle is not None:
                    all_handle.write(json.dumps(pair) + "\n")
                if cross_handle is not None and cross_split:
                    cross_handle.write(json.dumps(pair) + "\n")

    if all_handle is not None:
        all_handle.close()
    if cross_handle is not None:
        cross_handle.close()
    pairs.sort(key=lambda p: (p["hamming"], p["image_a"], p["image_b"]))
    representative = pairs[:representative_limit]
    cross = sum(1 for p in pairs if p["cross_split"])
    return {
        "method": "banded_ahash_lsh",
        "band_prefix_hex_chars": band_chars,
        "hamming_threshold": threshold,
        "images_with_phash": len(entries),
        "bands": len(bands),
        "candidate_pairs": len(pairs),
        "cross_split_candidate_pairs": cross,
        "hamming_histogram": {str(k): v for k, v in sorted(hamming_histogram.items())},
        "cross_split_hamming_histogram": {str(k): v for k, v in sorted(cross_split_histogram.items())},
        "representative_pairs": representative,
        "complete": len(entries) == len(rows),
        "documentation": {
            "hash": f"average hash {deep_get(load_config(), 'duplicates', 'near_hash_size', default=8)}x",
            "grouping_decision": (
                "Near-duplicate pairs are saved for review and do not automatically "
                "merge split groups. Exact SHA-256 duplicates and heuristic/specimen "
                "groups remain the leakage-prevention units."
            ),
        },
    }


def write_class_mapping(manifests_dir: Path) -> Path:
    mapping = {"Fresh": 0, "Spoiled": 1, "num_classes": 2}
    path = manifests_dir / "class_mapping.json"
    path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    return path


def ratio_deviation(counts: dict[str, int], targets: dict[str, float]) -> dict[str, Any]:
    total = sum(counts.values()) or 1
    actual = {k: counts.get(k, 0) / total for k in ("train", "val", "test")}
    return {
        "target": targets,
        "actual": {k: round(v, 6) for k, v in actual.items()},
        "absolute_deviation": {k: round(abs(actual[k] - targets.get(k, 0.0)), 6) for k in actual},
        "explanation": (
            "Deviations are expected under group-aware splitting: whole groups are "
            "assigned together to prevent leakage, so exact 70/15/15 image ratios "
            "are not forced."
        ),
    }


def load_near_scan_provenance() -> dict[str, Any] | None:
    if not NEAR_SCAN_PROVENANCE.is_file():
        return None
    return json.loads(NEAR_SCAN_PROVENANCE.read_text(encoding="utf-8"))


def load_near_review_status() -> dict[str, Any]:
    if not NEAR_REVIEW_STATUS.is_file():
        return {"complete": False, "status": "not_reviewed"}
    return json.loads(NEAR_REVIEW_STATUS.read_text(encoding="utf-8"))


def write_near_scan_provenance(payload: dict[str, Any]) -> None:
    NEAR_SCAN_PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    NEAR_SCAN_PROVENANCE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_integrity_audit(
    *,
    manifests_dir: Path | None = None,
    config: DatasetConfig | None = None,
    run_near_scan: bool = True,
    resume_hash_cache: bool = True,
    write_report: bool = True,
) -> dict[str, Any]:
    config = config or load_config()
    manifests_dir = manifests_dir or config.manifests_dir
    rows = load_manifest_rows(manifests_dir)
    counts = count_images_and_groups(rows)

    exact_leaks = find_cross_split_groups(
        rows, key_fn=lambda r: f"exact:{r.content_sha256}" if r.content_sha256 else r.image_path
    )
    # Only count real multi-image exact groups.
    exact_leaks = [g for g in exact_leaks if g["group_key"].startswith("exact:") and g["n_images"] >= 2]
    split_group_leaks = find_cross_split_groups(rows, key_fn=stable_split_group_key)

    cache_path = REPO_ROOT / "data" / "processed" / "cache" / "image_hashes.jsonl"
    near_summary: dict[str, Any]
    if run_near_scan:
        cache = ensure_hashes(
            rows,
            cache_path,
            hash_size=int(deep_get(config, "duplicates", "near_hash_size", default=8)),
            resume=resume_hash_cache,
        )
        near_summary = scalable_near_candidates(
            rows,
            cache,
            band_chars=3,
            threshold=int(deep_get(config, "duplicates", "near_hamming_threshold", default=5)),
            max_candidates_per_image=64,
            representative_limit=200,
        )
        pairs_path = REPO_ROOT / "data" / "processed" / "reports" / "near_duplicate_candidates.csv"
        pairs_path.parent.mkdir(parents=True, exist_ok=True)
        with pairs_path.open("w", newline="", encoding="utf-8") as handle:
            fields = [
                "image_a",
                "image_b",
                "hamming",
                "label_a",
                "label_b",
                "split_a",
                "split_b",
                "cross_split",
                "same_label",
                "note",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for pair in near_summary["representative_pairs"]:
                writer.writerow(pair)
        near_summary["representative_pairs_csv"] = str(pairs_path)
        fingerprint_path = manifests_dir / "split_fingerprint.json"
        fingerprint = None
        if fingerprint_path.is_file():
            fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8")).get("split_fingerprint")
        write_near_scan_provenance(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "dataset_fingerprint": fingerprint,
                "hash_cache_path": str(cache_path),
                "images_with_phash": near_summary.get("images_with_phash"),
                "candidate_pairs": near_summary.get("candidate_pairs"),
                "cross_split_candidate_pairs": near_summary.get("cross_split_candidate_pairs"),
                "candidate_search_complete": bool(near_summary.get("complete")),
                "hash_computation_complete": near_summary.get("images_with_phash") == len(rows),
                "scan_configuration": {
                    "method": near_summary.get("method"),
                    "hash": "average hash",
                    "hash_size": int(deep_get(config, "duplicates", "near_hash_size", default=8)),
                    "band_prefix_hex_chars": near_summary.get("band_prefix_hex_chars"),
                    "hamming_threshold": near_summary.get("hamming_threshold"),
                    "max_candidates_per_image": int(
                        deep_get(config, "duplicates", "near_max_candidates_per_image", default=64)
                    ),
                    "note": (
                        "Banded average-hash search with a per-image comparison cap. "
                        "Not an all-pairs comparison. Hamming distance is not proof of duplication."
                    ),
                },
                "representative_pairs_csv": str(pairs_path),
                "review_status_path": str(NEAR_REVIEW_STATUS),
            }
        )
    else:
        preserved = load_near_scan_provenance()
        if preserved and preserved.get("candidate_search_complete"):
            near_summary = {
                "complete": True,
                "skipped": True,
                "preserved_completed_scan": True,
                "provenance_path": str(NEAR_SCAN_PROVENANCE),
                "images_with_phash": preserved.get("images_with_phash"),
                "candidate_pairs": preserved.get("candidate_pairs"),
                "cross_split_candidate_pairs": preserved.get("cross_split_candidate_pairs"),
                "note": (
                    "Near-duplicate scan was not re-run. The completed hash and candidate-search "
                    "provenance was preserved and was not replaced with a skipped status."
                ),
            }
        else:
            near_summary = {
                "complete": False,
                "skipped": True,
                "preserved_completed_scan": False,
                "note": "Near-duplicate scan skipped by caller. No completed scan provenance file was found.",
            }

    targets = config.split_ratios
    split_counts = counts["images_by_split"]
    exact_ok = len(exact_leaks) == 0
    groups_ok = len(split_group_leaks) == 0
    near_complete = bool(near_summary.get("complete"))
    review_status = load_near_review_status()
    has_specimen_metadata = any(r.specimen_id or r.capture_session for r in rows)
    # Near scan completion removes the truncated-scan caveat, but without specimen /
    # capture metadata a residual leakage risk remains. Do not claim absolute
    # leakage-free final evaluation in that case.
    review_complete = bool(review_status.get("complete"))
    leakage_confirmed = bool(review_status.get("leakage_confirmed"))
    leakage_free_claim_allowed = (
        exact_ok and groups_ok and near_complete and review_complete and not leakage_confirmed and has_specimen_metadata
    )
    provisional = (
        not (exact_ok and groups_ok and near_complete)
        or not review_complete
        or leakage_confirmed
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifests_dir": str(manifests_dir),
        "counts": counts,
        "split_ratio_audit": ratio_deviation(split_counts, targets),
        "exact_duplicate_cross_split_groups": len(exact_leaks),
        "exact_duplicate_leak_examples": exact_leaks[:20],
        "split_group_cross_split_groups": len(split_group_leaks),
        "split_group_leak_examples": split_group_leaks[:20],
        "near_duplicate_review": near_summary,
        "near_duplicate_human_review": review_status,
        "integrity_status": {
            "exact_duplicates_separated": exact_ok,
            "known_groups_separated": groups_ok,
            "near_duplicate_scan_complete": near_complete,
            "near_duplicate_review_complete": review_complete,
            "cross_split_leakage_confirmed_by_review": leakage_confirmed,
            "specimen_or_session_metadata_present": has_specimen_metadata,
            "leakage_free_final_evaluation_claim_allowed": leakage_free_claim_allowed,
            "provisional_experiments_only": provisional,
            "residual_leakage_risk": not has_specimen_metadata,
            "notes": [
                "Near-duplicate perceptual matches are review candidates, not automatic specimen identity.",
                "No specimen_id/capture_session metadata is available on this dataset copy."
                if not has_specimen_metadata
                else "Specimen/session metadata present for grouping.",
                (
                    "If near scanning is incomplete, mark evaluation as provisional and "
                    "do not claim leakage-free final results."
                ),
                (
                    "Even with a complete near scan, heuristic grouping without specimen IDs "
                    "leaves residual leakage risk across differently named related captures."
                    if not has_specimen_metadata
                    else "Specimen-aware grouping is available."
                ),
            ],
        },
    }

    if not write_report:
        return report
    write_class_mapping(manifests_dir)
    out = REPO_ROOT / "data" / "processed" / "reports" / "integrity_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = out.with_suffix(".md")
    md.write_text(_integrity_markdown(report), encoding="utf-8")
    report["report_path"] = str(out)
    report["report_markdown_path"] = str(md)
    return report


def _integrity_markdown(report: dict[str, Any]) -> str:
    status = report["integrity_status"]
    lines = [
        "# FreshNet integrity audit",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Split counts",
        "",
        f"- Images by split: `{report['counts']['images_by_split']}`",
        f"- Unique groups by split: `{report['counts']['unique_groups_by_split']}`",
        f"- Images by split/class: `{report['counts']['images_by_split_and_class']}`",
        f"- Unique groups by split/class: `{report['counts']['unique_groups_by_split_and_class']}`",
        "",
        "## Leakage checks",
        "",
        f"- Exact duplicate groups crossing splits: **{report['exact_duplicate_cross_split_groups']}**",
        f"- Known split-groups crossing splits: **{report['split_group_cross_split_groups']}**",
        f"- Near-duplicate scan complete: **{status['near_duplicate_scan_complete']}**",
        f"- Leakage-free final evaluation claim allowed: **{status['leakage_free_final_evaluation_claim_allowed']}**",
        f"- Provisional experiments only: **{status['provisional_experiments_only']}**",
        "",
        "## Near-duplicate policy",
        "",
        report.get("near_duplicate_review", {}).get("documentation", {}).get(
            "grouping_decision",
            "Near matches are review candidates only.",
        ),
        "",
        "## Ratio deviation",
        "",
        report["split_ratio_audit"]["explanation"],
        f"- Target: `{report['split_ratio_audit']['target']}`",
        f"- Actual: `{report['split_ratio_audit']['actual']}`",
        "",
    ]
    return "\n".join(lines) + "\n"

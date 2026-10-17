#!/usr/bin/env python3
"""Build the versioned produce-classification manifests (metadata only).

Reads the frozen v3 source-group manifests (train/val/test) and writes
``data/processed/manifests_produce_v1/``. No image is decoded, no model is
touched, and the originals are never written. Produce type comes from the image
folder name; identity is independent of Fresh/Spoiled (FreshBanana and
RottenBanana are both ``banana``).

Run: backend/.venv/bin/python ml/produce/build_manifests.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "data" / "processed" / "manifests_v3_source"
OUT_DIR = ROOT / "data" / "processed" / "manifests_produce_v1"
DOCS_JSON = ROOT / "docs" / "produce_dataset_v1.json"
DOCS_MD = ROOT / "docs" / "PRODUCE_DATASET_V1.md"
SPLITS = ("train", "val", "test")
MANIFEST_VERSION = "freshnet-produce-manifests-v1"

# Canonical class -> display label and the folder spellings (lower-case, with the
# Fresh/Rotten prefix removed) that map to it. Evidence for each merge is in
# docs/PRODUCE_DATASET_V1.md. The canonical ids match the guidance catalogue.
CANONICAL = {
    "apple": {"label": "Apple", "aliases": ["apple"]},
    "banana": {"label": "Banana", "aliases": ["banana"]},
    "bitter_gourd": {"label": "Bitter gourd", "aliases": ["bittergroud"]},
    "capsicum": {
        "label": "Bell pepper / capsicum",
        "aliases": ["bellpepper", "capciscum", "capsicum"],
    },
    "carrot": {"label": "Carrot", "aliases": ["carrot"]},
    "cucumber": {"label": "Cucumber", "aliases": ["cucumber"]},
    "mango": {"label": "Mango", "aliases": ["mango"]},
    "okra": {"label": "Okra", "aliases": ["okara", "okra"]},
    "orange": {"label": "Orange", "aliases": ["orange"]},
    "potato": {"label": "Potato", "aliases": ["potato"]},
    "strawberry": {"label": "Strawberry", "aliases": ["strawberry"]},
    "tomato": {"label": "Tomato", "aliases": ["tomato"]},
}
ALIAS_TO_CLASS = {alias: cid for cid, spec in CANONICAL.items() for alias in spec["aliases"]}
CLASS_ORDER = sorted(CANONICAL)  # alphabetical, fixed: index = position
FOLDER_RE = re.compile(r"^(Fresh|Rotten)(.+)$")

MIXED_IMAGE_NOTE = (
    "Visual review: the picture shows a carrot, a cucumber and a cabbage side by side. "
    "The same pixels are filed under RottenBellpepper and RottenCarrot. It contains no "
    "bell pepper, and it is not a single-item image of any one class."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return path


def read_split(name: str) -> list[dict[str, str]]:
    with (SOURCE_DIR / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    source_hashes_before = {p.name: sha256_file(p) for p in sorted(SOURCE_DIR.glob("*")) if p.is_file()}
    source_fp = json.loads((SOURCE_DIR / "split_fingerprint.json").read_text())["split_fingerprint"]

    rows: list[dict[str, str]] = []
    for split in SPLITS:
        for row in read_split(split):
            row["split"] = split
            rows.append(row)

    # ---- map folders to canonical classes -------------------------------------------
    unmapped: Counter = Counter()
    for row in rows:
        match = FOLDER_RE.match(row["food_type"])
        if not match:
            unmapped[row["food_type"]] += 1
            continue
        prefix, name = match.group(1), match.group(2).lower()
        cid = ALIAS_TO_CLASS.get(name)
        if cid is None:
            unmapped[row["food_type"]] += 1
            continue
        row["_class"] = cid
        row["_alias"] = match.group(2)
        row["_prefix_agrees"] = str(
            (prefix == "Fresh" and row["label"] == "Fresh") or (prefix == "Rotten" and row["label"] == "Spoiled")
        )
    if unmapped:
        raise SystemExit(f"Folders with no canonical class: {dict(unmapped)}")
    if any(r["_prefix_agrees"] != "True" for r in rows):
        raise SystemExit("A folder prefix disagrees with the freshness label; investigate before building.")

    # ---- exclusions -----------------------------------------------------------------
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    excluded: dict[str, dict[str, str]] = {}
    for gid, members in groups.items():
        classes = {m["_class"] for m in members}
        if len(classes) > 1:
            for member in members:
                excluded[member["image_path"]] = {
                    "image_path": member["image_path"],
                    "relative_path": rel(member["image_path"]),
                    "source_folder": member["food_type"],
                    "split": member["split"],
                    "group_id": gid,
                    "reason": "multi_produce_source_group",
                    "detail": MIXED_IMAGE_NOTE
                    + f" Classes in this source group: {sorted(classes)}. All members excluded rather "
                    "than assigning an arbitrary class.",
                }
    by_hash: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_hash[row["content_sha256"]].add(row["_class"])
    cross_hash = [h for h, c in by_hash.items() if len(c) > 1]
    for row in rows:
        if row["content_sha256"] in cross_hash and row["image_path"] not in excluded:
            excluded[row["image_path"]] = {
                "image_path": row["image_path"],
                "relative_path": rel(row["image_path"]),
                "source_folder": row["food_type"],
                "split": row["split"],
                "group_id": row["group_id"],
                "reason": "identical_bytes_under_different_classes",
                "detail": "Same content hash appears under more than one canonical class.",
            }

    kept = [r for r in rows if r["image_path"] not in excluded]

    # ---- integrity checks -----------------------------------------------------------
    group_splits: dict[str, set[str]] = defaultdict(set)
    group_class: dict[str, set[str]] = defaultdict(set)
    for row in kept:
        group_splits[row["group_id"]].add(row["split"])
        group_class[row["group_id"]].add(row["_class"])
    crossing = [g for g, s in group_splits.items() if len(s) > 1]
    mixed = [g for g, c in group_class.items() if len(c) > 1]
    missing_files = [r["image_path"] for r in kept if not Path(r["image_path"]).is_file()]
    if crossing or mixed or missing_files:
        raise SystemExit(f"Integrity failure: crossing={len(crossing)} mixed={len(mixed)} missing={len(missing_files)}")
    alias_spanning_groups = [
        g
        for g, members in groups.items()
        if len({m["_alias"].lower() for m in members if m["_class"] == "capsicum"}) > 1
    ]

    # ---- write manifests ------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mapping = {
        "manifest_version": MANIFEST_VERSION,
        "task": "produce_type_identification",
        "num_classes": len(CLASS_ORDER),
        "label_order": CLASS_ORDER,
        "classes": {cid: i for i, cid in enumerate(CLASS_ORDER)},
        "display_labels": {cid: CANONICAL[cid]["label"] for cid in CLASS_ORDER},
        "aliases": {
            cid: {
                "folder_spellings": sorted({r["_alias"] for r in rows if r["_class"] == cid}),
                "alias_keys": CANONICAL[cid]["aliases"],
            }
            for cid in CLASS_ORDER
        },
        "note": "Identity is independent of Fresh/Spoiled. There is no unknown/non-produce class.",
    }
    (OUT_DIR / "class_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    fields = [
        "image_path",
        "relative_path",
        "produce",
        "produce_index",
        "freshness_label",
        "source_folder",
        "split",
        "group_id",
        "content_sha256",
        "width",
        "height",
    ]
    out_rows: dict[str, list[dict[str, str]]] = {s: [] for s in SPLITS}
    for row in kept:
        out_rows[row["split"]].append(
            {
                "image_path": row["image_path"],
                "relative_path": rel(row["image_path"]),
                "produce": row["_class"],
                "produce_index": str(CLASS_ORDER.index(row["_class"])),
                "freshness_label": row["label"],
                "source_folder": row["food_type"],
                "split": row["split"],
                "group_id": row["group_id"],
                "content_sha256": row["content_sha256"],
                "width": row.get("width", ""),
                "height": row.get("height", ""),
            }
        )
    for split in SPLITS:
        with (OUT_DIR / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(out_rows[split])
    with (OUT_DIR / "all.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split in SPLITS:
            writer.writerows(out_rows[split])
    ex_fields = ["image_path", "relative_path", "source_folder", "split", "group_id", "reason", "detail"]
    with (OUT_DIR / "exclusions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ex_fields)
        writer.writeheader()
        writer.writerows(sorted(excluded.values(), key=lambda r: r["relative_path"]))

    # ---- counts + fingerprint -------------------------------------------------------
    counts = {s: Counter(r["produce"] for r in out_rows[s]) for s in SPLITS}
    freshness_counts = {
        s: {c: dict(Counter(r["freshness_label"] for r in out_rows[s] if r["produce"] == c)) for c in CLASS_ORDER}
        for s in SPLITS
    }
    group_counts = {
        s: {c: len({r["group_id"] for r in out_rows[s] if r["produce"] == c}) for c in CLASS_ORDER} for s in SPLITS
    }
    canonical_rows = sorted(
        (r["relative_path"], r["split"], r["produce"], r["group_id"], r["content_sha256"])
        for s in SPLITS
        for r in out_rows[s]
    )
    fp_payload = json.dumps(
        {
            "manifest_version": MANIFEST_VERSION,
            "source_split_fingerprint": source_fp,
            "class_order": CLASS_ORDER,
            "rows": canonical_rows,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha256(fp_payload.encode("utf-8")).hexdigest()

    source_hashes_after = {p.name: sha256_file(p) for p in sorted(SOURCE_DIR.glob("*")) if p.is_file()}
    if source_hashes_before != source_hashes_after:
        raise SystemExit("Source manifests changed during the build. This must never happen.")

    report = {
        "manifest_version": MANIFEST_VERSION,
        "produce_fingerprint": fingerprint,
        "source_split_fingerprint": source_fp,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifests_unchanged": True,
        "source_manifest_sha256": source_hashes_after,
        "num_classes": len(CLASS_ORDER),
        "classes": CLASS_ORDER,
        "included_images": {s: len(out_rows[s]) for s in SPLITS},
        "included_total": sum(len(out_rows[s]) for s in SPLITS),
        "excluded_total": len(excluded),
        "exclusion_reasons": dict(Counter(e["reason"] for e in excluded.values())),
        "counts_by_split": {s: {c: counts[s].get(c, 0) for c in CLASS_ORDER} for s in SPLITS},
        "freshness_counts_by_split": freshness_counts,
        "source_group_counts_by_split": group_counts,
        "source_groups_crossing_splits": len(crossing),
        "source_groups_with_more_than_one_class": len(mixed),
        "groups_spanning_capsicum_folder_spellings": len(alias_spanning_groups),
        "identical_bytes_across_classes": len(cross_hash),
        "alias_map": mapping["aliases"],
        "exclusions": sorted(excluded.values(), key=lambda r: r["relative_path"]),
        "limits": [
            "Produce type is the dataset folder name. Spot-checked visually on train samples, not verified image by image.",
            "No specimen ids. Counts are images, not distinct specimens, and many are augmented copies.",
            "There are no non-produce or out-of-scope images, so no unknown class exists or can be evaluated.",
        ],
    }
    (OUT_DIR / "produce_fingerprint.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    DOCS_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    DOCS_MD.write_text(render_markdown(report, mapping), encoding="utf-8")
    keys = ("produce_fingerprint", "included_images", "excluded_total", "exclusion_reasons")
    print(json.dumps({k: report[k] for k in keys}, indent=2))
    return 0


def render_markdown(report: dict, mapping: dict) -> str:
    classes = report["classes"]
    lines = [
        "# Produce classification dataset v1",
        "",
        "Generated by `ml/produce/build_manifests.py` from the frozen v3 source-group manifests. Metadata only: no image was decoded, no model was touched, and the v3 manifests and image files were not modified (their SHA-256 values were checked before and after).",
        "",
        f"- Manifest version: `{report['manifest_version']}`",
        f"- **Produce dataset fingerprint:** `{report['produce_fingerprint']}`",
        f"- Source (v3) split fingerprint: `{report['source_split_fingerprint']}`",
        "- Location: `data/processed/manifests_produce_v1/` (train, val, test, all, exclusions, class_mapping, produce_fingerprint)",
        f"- Included images: {report['included_total']:,} (train {report['included_images']['train']:,}, val {report['included_images']['val']:,}, test {report['included_images']['test']:,})",
        f"- Excluded images: {report['excluded_total']}",
        "",
        "Train/validation/test assignments and source-group boundaries are exactly those of v3. No source group crosses a split, and no kept source group contains more than one produce class.",
        "",
        "## Classes and aliases",
        "",
        "Produce identity does not depend on Fresh/Spoiled: `FreshBanana` and `RottenBanana` are both `banana`. The 12 supported classes are the ones with enough labelled images. There is no Unknown or non-produce class, because the data has no such images.",
        "",
        "| Index | Class id | Label | Folder spellings merged into it |",
        "| ---: | --- | --- | --- |",
    ]
    for i, cid in enumerate(classes):
        lines.append(
            f"| {i} | `{cid}` | {mapping['display_labels'][cid]} | {', '.join(mapping['aliases'][cid]['folder_spellings'])} |"
        )
    lines += [
        "",
        "### Why the aliases were merged",
        "",
        "- **Bell pepper + Capsicum (`capsicum`).** All folder families (`Bellpepper`, `Capciscum`, `Capsicum`) hold the same thing: blocky bell peppers. `Bellpepper` is a mixed set of web photos in several colours; `Capciscum` (Fresh) and `Capsicum` (Rotten) are one green-pepper set photographed on a white background, many with rotation-augmentation corners. Looking at random train images from the four folders found no separate category, so they are one class. Caveat: a few `RottenBellpepper` web images show elongated chili-type peppers, and that noise is not removed. The canonical id `capsicum` is the one already used by the API and guidance catalogue.",
        "- **Okara + Okra (`okra`).** `FreshOkara` and `RottenOkra` are one produce type. The `Okara` images are okra pods (not the soy pulp that shares the name). This spelling split also means all train Fresh okra comes from one folder and all train Spoiled okra from another.",
        "- **Bittergroud (`bitter_gourd`).** A misspelling of bitter gourd. The images show bitter gourds.",
        "",
        "## Counts by split (included images)",
        "",
        "| Class | Train | Val | Test | Total |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for cid in classes:
        t, v, e = (report["counts_by_split"][s][cid] for s in SPLITS)
        lines.append(f"| {cid} | {t:,} | {v:,} | {e:,} | {t + v + e:,} |")
    lines += ["", "## Source groups per class and split", "", "| Class | Train | Val | Test |", "| --- | ---: | ---: | ---: |"]
    for cid in classes:
        g = report["source_group_counts_by_split"]
        lines.append(f"| {cid} | {g['train'][cid]:,} | {g['val'][cid]:,} | {g['test'][cid]:,} |")
    lines += [
        "",
        "## Fresh and Spoiled mix inside each class",
        "",
        "Used only for the subgroup report (does a spoiled look hurt identity recognition?). It is not a model input.",
        "",
        "| Class | Train Fresh / Spoiled | Val Fresh / Spoiled | Test Fresh / Spoiled |",
        "| --- | --- | --- | --- |",
    ]
    for cid in classes:
        cells = []
        for s in SPLITS:
            f = report["freshness_counts_by_split"][s][cid]
            cells.append(f"{f.get('Fresh', 0):,} / {f.get('Spoiled', 0):,}")
        lines.append(f"| {cid} | " + " | ".join(cells) + " |")
    lines += ["", "## Exclusions", ""]
    for ex in report["exclusions"]:
        lines.append(f"- `{ex['relative_path']}` ({ex['split']}, folder {ex['source_folder']}): {ex['reason']}")
    lines += [
        "",
        MIXED_IMAGE_NOTE,
        "Its source group contained exactly these two files (both train), so no other ambiguous sibling exists. All other source groups contain a single produce class.",
        "",
        "## Limits",
        "",
    ] + [f"- {item}" for item in report["limits"]]
    lines.append(
        "- The same v3 test images were used by the frozen freshness model. A produce evaluation on them is a separate-task evaluation on the existing holdout, not a newly collected external test set."
    )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())

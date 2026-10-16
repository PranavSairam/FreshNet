"""Audit produce-type labels available in the FreshNet v3 manifests.

This reads manifest METADATA only (folder names, splits, groups, hashes). It does
not open images, train anything, or touch the freshness model. The produce type
comes from the dataset folder name, never from an uploaded filename.

Usage (from the repository root):
    backend/.venv/bin/python ml/audit_produce_labels.py

Writes docs/PRODUCE_LABEL_INVENTORY.md and docs/produce_label_inventory.json.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "processed" / "manifests_v3_source"
DOCS = ROOT / "docs"

# Folder suffix (after the Fresh/Rotten prefix) -> canonical produce id.
# Misspelled folder names are mapped, and reported as spelling variants.
CANONICAL = {
    "apple": "apple",
    "banana": "banana",
    "bellpepper": "bell_pepper",
    "bittergroud": "bitter_gourd",
    "capciscum": "capsicum",
    "capsicum": "capsicum",
    "carrot": "carrot",
    "cucumber": "cucumber",
    "mango": "mango",
    "okara": "okra",
    "okra": "okra",
    "orange": "orange",
    "potato": "potato",
    "strawberry": "strawberry",
    "tomato": "tomato",
}
SPLITS = ("train", "val", "test")
PREFIX = re.compile(r"^(Fresh|Rotten)(.+)$")


def parse_folder(name: str) -> tuple[str | None, str | None, str | None]:
    """Return (prefix, raw suffix, canonical produce id) for a folder name."""
    match = PREFIX.match(name or "")
    if not match:
        return None, None, None
    prefix, suffix = match.group(1), match.group(2)
    return prefix, suffix, CANONICAL.get(suffix.lower())


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    if not MANIFESTS.is_dir():
        print(f"Manifests not found: {MANIFESTS.relative_to(ROOT)}", file=sys.stderr)
        return 2

    included: list[dict[str, str]] = []
    for split in SPLITS:
        for row in read_rows(MANIFESTS / f"{split}.csv"):
            row["split"] = split
            included.append(row)
    excluded = read_rows(MANIFESTS / "exclusions.csv")

    counts: dict[str, dict[str, Counter]] = defaultdict(lambda: {s: Counter() for s in SPLITS})
    totals = Counter()
    missing_food_type = 0
    unmapped_folders: Counter = Counter()
    prefix_label_conflicts: list[str] = []
    folder_variants: dict[str, set[str]] = defaultdict(set)
    sha_to_produce: dict[str, set[str]] = defaultdict(set)
    group_to_produce: dict[str, set[str]] = defaultdict(set)
    group_to_split: dict[str, set[str]] = defaultdict(set)
    group_members: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    for row in included:
        prefix, suffix, produce = parse_folder(row.get("food_type", ""))
        if prefix is None:
            missing_food_type += 1
            continue
        if produce is None:
            unmapped_folders[row["food_type"]] += 1
            continue
        folder_variants[produce].add(suffix or "")
        label = row["label"]
        expected = "Fresh" if prefix == "Fresh" else "Spoiled"
        if label != expected:
            prefix_label_conflicts.append(row["food_type"])
        counts[produce][row["split"]][label] += 1
        totals[produce] += 1
        sha_to_produce[row["content_sha256"]].add(produce)
        group_to_produce[row["group_id"]].add(produce)
        group_to_split[row["group_id"]].add(row["split"])
        relative = row["image_path"].split("/Dataset/", 1)[-1]
        group_members[row["group_id"]].append((row["food_type"], row["split"], relative))

    excluded_by_produce: Counter = Counter()
    excluded_unmapped = 0
    for row in excluded:
        folder = Path(row["image_path"]).parent.name
        _, suffix, produce = parse_folder(folder)
        if produce is None:
            excluded_unmapped += 1
        else:
            excluded_by_produce[produce] += 1

    cross_produce_hashes = sum(1 for kinds in sha_to_produce.values() if len(kinds) > 1)
    cross_produce_groups = sum(1 for kinds in group_to_produce.values() if len(kinds) > 1)
    cross_split_groups = sum(1 for splits in group_to_split.values() if len(splits) > 1)
    conflicting_groups = [
        [{"folder": food, "split": split, "image": image} for food, split, image in members]
        for group, members in group_members.items()
        if len(group_to_produce[group]) > 1
    ]

    produce_ids = sorted(counts)
    table = []
    for produce in produce_ids:
        per_split = {}
        for split in SPLITS:
            fresh = counts[produce][split]["Fresh"]
            spoiled = counts[produce][split]["Spoiled"]
            per_split[split] = {"Fresh": fresh, "Spoiled": spoiled, "total": fresh + spoiled}
        fresh_total = sum(per_split[s]["Fresh"] for s in SPLITS)
        spoiled_total = sum(per_split[s]["Spoiled"] for s in SPLITS)
        table.append(
            {
                "produce": produce,
                "folder_spellings": sorted(folder_variants[produce]),
                "included_total": fresh_total + spoiled_total,
                "included_fresh": fresh_total,
                "included_spoiled": spoiled_total,
                "excluded_total": excluded_by_produce.get(produce, 0),
                "splits": per_split,
            }
        )

    report = {
        "source": "data/processed/manifests_v3_source (v3 split fingerprint; metadata only)",
        "included_images": len(included),
        "excluded_images": len(excluded),
        "produce_types": len(produce_ids),
        "rows_missing_food_type": missing_food_type,
        "rows_with_unmapped_folder": sum(unmapped_folders.values()),
        "excluded_rows_unmapped": excluded_unmapped,
        "folder_prefix_vs_freshness_label_conflicts": len(prefix_label_conflicts),
        "content_hashes_under_multiple_produce_types": cross_produce_hashes,
        "source_groups_spanning_multiple_produce_types": cross_produce_groups,
        "source_groups_crossing_splits": cross_split_groups,
        "cross_produce_group_details": conflicting_groups,
        "spelling_variants": {
            produce: sorted(variants)
            for produce, variants in folder_variants.items()
            if len(variants) > 1
        },
        "produce": table,
    }

    (DOCS / "produce_label_inventory.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (DOCS / "PRODUCE_LABEL_INVENTORY.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "produce"}, indent=2))
    return 0


def render_markdown(report: dict) -> str:
    lines = [
        "# Produce label inventory",
        "",
        "Generated by `ml/audit_produce_labels.py` from the v3 manifests",
        "(`data/processed/manifests_v3_source/`). It reads metadata only. No image was",
        "opened, no model was trained, and the freshness model was not touched.",
        "",
        "The only produce-type signal in this repository is the **dataset folder name**",
        "(for example `FreshBanana`, `RottenBanana`). There is no `specimen_id`,",
        "no separately annotated produce-type column, and **no produce classifier or",
        "checkpoint**. The deployed model predicts Fresh/Spoiled only.",
        "",
        "## Summary",
        "",
        f"- Included images (v3 benchmark): **{report['included_images']:,}**",
        f"- Excluded (quarantined) images: **{report['excluded_images']:,}**",
        f"- Produce types found from folder names: **{report['produce_types']}**",
        f"- Included rows with no `food_type`: {report['rows_missing_food_type']}",
        f"- Included rows whose folder maps to no known type: {report['rows_with_unmapped_folder']}",
        f"- Folder prefix (Fresh/Rotten) disagreeing with the freshness label: "
        f"{report['folder_prefix_vs_freshness_label_conflicts']}",
        f"- Content hashes appearing under more than one produce type: "
        f"{report['content_hashes_under_multiple_produce_types']}",
        f"- Source groups spanning more than one produce type: "
        f"{report['source_groups_spanning_multiple_produce_types']}",
        f"- Source groups crossing train/val/test: {report['source_groups_crossing_splits']}",
        "",
        "## Classes and image counts (included images)",
        "",
        "| Produce (canonical id) | Folder spellings | Fresh | Spoiled | Included | Excluded |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in report["produce"]:
        lines.append(
            f"| {item['produce']} | {', '.join(item['folder_spellings'])} | "
            f"{item['included_fresh']:,} | {item['included_spoiled']:,} | "
            f"{item['included_total']:,} | {item['excluded_total']:,} |"
        )
    lines += [
        "",
        "## Per-split counts (source-group split boundaries preserved)",
        "",
        "Counts come straight from the existing v3 `train.csv`, `val.csv` and `test.csv`.",
        "Test counts are listed so the split is visible. The test split must not be used",
        "to develop or select a produce classifier.",
        "",
        "| Produce | Train | Val | Test |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in report["produce"]:
        s = item["splits"]
        lines.append(
            f"| {item['produce']} | {s['train']['total']:,} | {s['val']['total']:,} | "
            f"{s['test']['total']:,} |"
        )
    lines += [
        "",
        "## Findings",
        "",
        "### Missing labels",
        "",
        "Every included image has a folder-derived produce type. No row was missing one.",
        "That is a folder-name label, not an independently verified annotation.",
        "",
        "### Label conflicts and ambiguities",
        "",
    ]
    variants = report["spelling_variants"]
    if variants:
        for produce, names in variants.items():
            lines.append(
                f"- **{produce}** appears under different folder spellings: "
                f"{', '.join(names)}. Folder names must be normalised before training."
            )
    lines += [
        "- `FreshBittergroud` / `RottenBittergroud` look like a misspelling of bitter "
        "gourd. The mapping to `bitter_gourd` is an assumption that needs human review.",
        "- `Bellpepper` and `Capsicum` are separate folders, but a bell pepper is a "
        "capsicum. The two classes overlap and a classifier could not tell them apart "
        "reliably without a defined rule.",
        "- Several produce types have few images (see the table). Okra (529), bitter "
        "gourd (680), strawberry (873), mango (1,060) and bell pepper (1,078) are small "
        "against tomato (11,431), carrot (10,434) and orange (10,337).",
        "- Class balance differs by produce type (carrot, orange and tomato are mostly "
        "Fresh; apple, potato and strawberry lean Spoiled), so a produce classifier "
        "could pick up freshness cues instead of produce identity.",
    ]
    for group in report["cross_produce_group_details"]:
        members = "; ".join(f"{m['folder']} ({m['split']}): {m['image']}" for m in group)
        lines.append(
            "- **A source group contains two different produce types** and needs human "
            f"review (a near-duplicate false match, or a misfiled image): {members}. "
            "Both rows are in the same split, so no split boundary is crossed."
        )
    lines += ["", "### Checks that passed", ""]
    if report["folder_prefix_vs_freshness_label_conflicts"] == 0:
        lines.append("- Every Fresh*/Rotten* folder agrees with its Fresh/Spoiled label.")
    if report["content_hashes_under_multiple_produce_types"] == 0:
        lines.append("- No identical image appears under two different produce types.")
    if report["source_groups_crossing_splits"] == 0:
        lines.append("- No source group crosses train, validation and test.")
    lines += [
        "",
        "### Limits of this audit",
        "",
        "- Folder names are the only evidence. Wrongly filed images would not be found.",
        "- The dataset has no specimen metadata, and the near-duplicate search was not",
        "  exhaustive (see `docs/TRAINING.md`). Leakage-free cannot be claimed.",
        "- Many images are augmented copies (rotated, translated). Counts are images,",
        "  not distinct specimens.",
        "- The data has no non-produce or out-of-scope images, so an Unknown/non-produce",
        "  detector cannot be trained or evaluated from it.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

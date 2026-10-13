#!/usr/bin/env python3
"""Create the tiny synthetic fixture used only by pipeline tests.

Marked as test data. Never use for real training or evaluation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "data" / "fixtures" / "synthetic_pipeline"


def _write_image(path: Path, color: tuple[int, int, int], mark: str, size: tuple[int, int] = (64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)
    draw.rectangle((8, 8, 24, 24), fill=(255, 255, 255))
    draw.text((28, 28), mark, fill=(0, 0, 0))
    img.save(path)


def main() -> int:
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    # Wipe previous generated images but keep README if present.
    for path in FIXTURE_ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".csv", ".json"}:
            path.unlink()

    meta = {
        "purpose": "synthetic_pipeline_test_fixture",
        "exclude_from_real_training": True,
        "exclude_from_real_evaluation": True,
        "note": "Not a real food dataset. Used only to test preparation code.",
    }
    (FIXTURE_ROOT / "FIXTURE_META.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Class-folder layout with two labels only (no Medium).
    fresh_a = FIXTURE_ROOT / "class_folders" / "Fresh" / "Apple"
    fresh_b = FIXTURE_ROOT / "class_folders" / "Fresh" / "Banana"
    spoiled = FIXTURE_ROOT / "class_folders" / "Rotten" / "Apple"

    _write_image(fresh_a / "specimen1_view1.png", (40, 160, 40), "F1")
    _write_image(fresh_a / "specimen1_view2.png", (42, 158, 40), "F1")  # near-duplicate of view1
    _write_image(fresh_b / "specimen2.png", (50, 170, 50), "F2")
    _write_image(spoiled / "specimen3.png", (120, 40, 40), "S1")

    # Exact duplicate copy
    exact_src = fresh_b / "specimen2.png"
    exact_dst = fresh_b / "specimen2_copy.png"
    exact_dst.write_bytes(exact_src.read_bytes())

    # Unreadable / invalid file
    bad = FIXTURE_ROOT / "class_folders" / "Fresh" / "Apple" / "corrupt.png"
    bad.write_text("not-an-image", encoding="utf-8")

    # CSV manifest with grouping metadata (includes Medium absence; only Fresh/Spoiled via Rotten mapping).
    csv_dir = FIXTURE_ROOT / "csv_mode"
    csv_dir.mkdir(parents=True, exist_ok=True)
    images = [
        ("img_a.png", (30, 150, 30), "Fresh", "apple", "specA", "sess1"),
        ("img_b.png", (32, 148, 30), "Fresh", "apple", "specA", "sess1"),
        ("img_c.png", (140, 50, 50), "Rotten", "apple", "specB", "sess2"),
        ("img_d.png", (145, 55, 55), "Spoiled", "banana", "specC", "sess3"),
    ]
    rows = []
    for name, color, label, food, specimen, session in images:
        path = csv_dir / "images" / name
        _write_image(path, color, specimen[:2])
        rows.append(
            {
                "image_path": str(path.relative_to(REPO_ROOT)),
                "label": label,
                "food_type": food,
                "specimen_id": specimen,
                "capture_session": session,
                "source": "synthetic_fixture",
                "fixture": "true",
            }
        )
    # Missing path row
    rows.append(
        {
            "image_path": str((csv_dir / "images" / "missing.png").relative_to(REPO_ROOT)),
            "label": "Fresh",
            "food_type": "apple",
            "specimen_id": "specMissing",
            "capture_session": "sessX",
            "source": "synthetic_fixture",
            "fixture": "true",
        }
    )
    # Invalid label
    bad_label_path = csv_dir / "images" / "bad_label.png"
    _write_image(bad_label_path, (10, 10, 10), "XX")
    rows.append(
        {
            "image_path": str(bad_label_path.relative_to(REPO_ROOT)),
            "label": "AlmostFresh",
            "food_type": "apple",
            "specimen_id": "specBad",
            "capture_session": "sessY",
            "source": "synthetic_fixture",
            "fixture": "true",
        }
    )

    manifest = csv_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "label",
                "food_type",
                "specimen_id",
                "capture_session",
                "source",
                "fixture",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    readme = FIXTURE_ROOT / "README.md"
    readme.write_text(
        "# Synthetic pipeline fixture\n\n"
        "TEST DATA ONLY. Excluded from real training and evaluation.\n"
        "Regenerate with: `python ml/dataset/build_fixture.py`\n",
        encoding="utf-8",
    )
    print(f"Wrote fixture at {FIXTURE_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

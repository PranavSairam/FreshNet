from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .config import DatasetConfig, deep_get
from .records import ImageRecord


def validate_records(records: list[ImageRecord], config: DatasetConfig) -> dict:
    min_side = int(deep_get(config, "validation", "min_side_px", default=16))
    max_sample = int(deep_get(config, "validation", "max_sample_for_dimension_stats", default=5000))

    missing_files = 0
    unreadable = 0
    invalid_labels = 0
    widths: list[int] = []
    heights: list[int] = []

    candidates = [r for r in records if Path(r.image_path).is_file()]
    sample = candidates
    if len(candidates) > max_sample:
        rng = random.Random(config.split_seed)
        sample = rng.sample(candidates, max_sample)
    sample_set = {id(r) for r in sample}

    for index, record in enumerate(records, start=1):
        if index == 1 or index % 2000 == 0 or index == len(records):
            print(f"[validate] {index}/{len(records)}", flush=True)
        path = Path(record.image_path)
        if not path.is_file():
            missing_files += 1
            record.readable = False
            record.problems.append("missing_file")
            continue

        if record.label is None:
            invalid_labels += 1
            record.problems.append("invalid_label")

        # Always try to open; dimension stats only retained for the sample.
        try:
            with Image.open(path) as img:
                img.load()
                width, height = img.size
                record.width = width
                record.height = height
                record.readable = True
                if width < min_side or height < min_side:
                    record.problems.append("too_small")
                if id(record) in sample_set:
                    widths.append(width)
                    heights.append(height)
        except (UnidentifiedImageError, OSError, ValueError):
            unreadable += 1
            record.readable = False
            record.problems.append("unreadable")

    label_counts = Counter(r.label for r in records if r.label is not None)
    total_labeled = sum(label_counts.values())
    imbalance = {}
    if total_labeled:
        for label, count in sorted(label_counts.items()):
            imbalance[label] = {
                "count": count,
                "fraction": round(count / total_labeled, 6),
            }

    dim_stats = {}
    if widths:
        dim_stats = {
            "sampled": len(widths),
            "width": {"min": min(widths), "max": max(widths), "mean": round(sum(widths) / len(widths), 2)},
            "height": {
                "min": min(heights),
                "max": max(heights),
                "mean": round(sum(heights) / len(heights), 2),
            },
        }

    return {
        "total_records": len(records),
        "missing_files": missing_files,
        "unreadable": unreadable,
        "invalid_labels": invalid_labels,
        "label_counts": dict(label_counts),
        "class_imbalance": imbalance,
        "dimension_stats": dim_stats,
        "readable": sum(1 for r in records if r.readable is True),
    }

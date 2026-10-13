from __future__ import annotations

from dataclasses import dataclass, field


INTENDED = ("Fresh", "Medium", "Spoiled")


@dataclass
class LabelResolution:
    canonical_labels: list[str]
    raw_label_counts: dict[str, int]
    mapped_counts: dict[str, int]
    unmapped_labels: list[str]
    has_fresh: bool
    has_medium: bool
    has_spoiled: bool
    three_class_ready: bool
    two_class_ready: bool
    limitation_notes: list[str] = field(default_factory=list)


def normalize_raw_label(value: str) -> str:
    return value.strip()


def map_label(raw_label: str, mapping: dict[str, str]) -> str | None:
    key = normalize_raw_label(raw_label)
    if key in mapping:
        return mapping[key]
    lowered = {k.lower(): v for k, v in mapping.items()}
    return lowered.get(key.lower())


def resolve_labels(
    raw_labels: list[str],
    mapping: dict[str, str],
    *,
    intended_classes: list[str] | None = None,
    require_for_three_class: list[str] | None = None,
    allow_two_class_training: bool = True,
) -> LabelResolution:
    intended = intended_classes or list(INTENDED)
    required = require_for_three_class or list(INTENDED)

    raw_counts: dict[str, int] = {}
    mapped_counts: dict[str, int] = {}
    unmapped: set[str] = set()

    for raw in raw_labels:
        raw_counts[raw] = raw_counts.get(raw, 0) + 1
        mapped = map_label(raw, mapping)
        if mapped is None:
            unmapped.add(raw)
            continue
        mapped_counts[mapped] = mapped_counts.get(mapped, 0) + 1

    present = sorted(mapped_counts)
    has_fresh = "Fresh" in mapped_counts
    has_medium = "Medium" in mapped_counts
    has_spoiled = "Spoiled" in mapped_counts
    three_class_ready = all(label in mapped_counts for label in required)
    two_class_ready = has_fresh and has_spoiled and not has_medium

    notes: list[str] = []
    if not has_medium:
        notes.append(
            "Medium is absent after label mapping. Three-class Fresh/Medium/Spoiled "
            "training is not allowed until a properly annotated dataset providing "
            "Medium labels is supplied. FreshNet will not manufacture Medium."
        )
    if two_class_ready and allow_two_class_training:
        notes.append(
            "Two-class training (Fresh vs Spoiled) is allowed with allow_two_class_training=true. "
            "Canonical labels used for training will be: Fresh, Spoiled."
        )
    if unmapped:
        notes.append(f"Unmapped raw labels: {sorted(unmapped)}")
    missing_intended = [label for label in intended if label not in mapped_counts]
    if missing_intended:
        notes.append(f"Intended application classes missing from data: {missing_intended}")

    if three_class_ready:
        canonical = list(required)
    elif two_class_ready and allow_two_class_training:
        canonical = ["Fresh", "Spoiled"]
    else:
        canonical = present

    return LabelResolution(
        canonical_labels=canonical,
        raw_label_counts=dict(sorted(raw_counts.items())),
        mapped_counts=dict(sorted(mapped_counts.items())),
        unmapped_labels=sorted(unmapped),
        has_fresh=has_fresh,
        has_medium=has_medium,
        has_spoiled=has_spoiled,
        three_class_ready=three_class_ready,
        two_class_ready=two_class_ready,
        limitation_notes=notes,
    )

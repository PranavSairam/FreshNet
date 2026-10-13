from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int = 2) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix


def normalize_confusion(matrix: list[list[int]]) -> list[list[float]]:
    out: list[list[float]] = []
    for row in matrix:
        total = sum(row) or 1
        out.append([round(v / total, 6) for v in row])
    return out


def per_class_scores(y_true: list[int], y_pred: list[int], num_classes: int = 2) -> dict[str, dict[str, float]]:
    matrix = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    names = {0: "Fresh", 1: "Spoiled"}
    scores: dict[str, dict[str, float]] = {}
    for cls in range(num_classes):
        tp = matrix[cls][cls]
        fp = sum(matrix[r][cls] for r in range(num_classes) if r != cls)
        fn = sum(matrix[cls][c] for c in range(num_classes) if c != cls)
        support = sum(matrix[cls])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        scores[names[cls]] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }
    return scores


def compute_classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true/y_pred must be non-empty and aligned.")
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    accuracy = correct / len(y_true)
    per_class = per_class_scores(y_true, y_pred)
    recalls = [v["recall"] for v in per_class.values()]
    balanced_accuracy = float(np.mean(recalls))
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))
    matrix = confusion_matrix(y_true, y_pred)
    # Spoiled=1 → Fresh=0 false negatives for spoilage detection.
    spoiled_total = sum(1 for t in y_true if t == 1)
    spoiled_to_fresh = matrix[1][0]
    spoiled_to_fresh_rate = (spoiled_to_fresh / spoiled_total) if spoiled_total else 0.0
    return {
        "n": len(y_true),
        "accuracy": round(accuracy, 6),
        "balanced_accuracy": round(balanced_accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "confusion_matrix_normalized": normalize_confusion(matrix),
        "spoiled_to_fresh_errors": spoiled_to_fresh,
        "spoiled_test_images": spoiled_total,
        "spoiled_to_fresh_rate": round(spoiled_to_fresh_rate, 6),
        "label_distribution_true": dict(Counter(y_true)),
        "label_distribution_pred": dict(Counter(y_pred)),
    }


def class_weights_from_labels(labels: list[int], num_classes: int = 2) -> list[float]:
    counts = Counter(labels)
    total = len(labels) or 1
    weights = []
    for cls in range(num_classes):
        c = counts.get(cls, 0)
        weights.append(total / (num_classes * c) if c else 0.0)
    return weights

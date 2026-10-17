"""Multi-class metrics for the produce classifier (numpy only)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        matrix[int(t), int(p)] += 1
    return matrix


def metrics_from_confusion(matrix: np.ndarray, class_names: Sequence[str]) -> dict[str, Any]:
    num_classes = matrix.shape[0]
    total = int(matrix.sum())
    if total == 0:
        raise ValueError("Empty confusion matrix.")
    per_class: dict[str, dict[str, float | int]] = {}
    recalls: list[float] = []
    f1s: list[float] = []
    present_f1s: list[float] = []
    for c in range(num_classes):
        tp = int(matrix[c, c])
        fp = int(matrix[:, c].sum() - tp)
        fn = int(matrix[c, :].sum() - tp)
        support = int(matrix[c, :].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[class_names[c]] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }
        if support:
            recalls.append(recall)
            present_f1s.append(f1)
        f1s.append(f1)
    return {
        "n": total,
        "accuracy": round(float(np.trace(matrix)) / total, 6),
        # Averages are over classes that have at least one true sample (all 12 on the real splits).
        "macro_f1": round(float(np.mean(present_f1s)), 6),
        "balanced_accuracy": round(float(np.mean(recalls)), 6),
        "per_class": per_class,
        "class_order": list(class_names),
        "confusion_matrix": matrix.tolist(),
    }


def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]
) -> dict[str, Any]:
    return metrics_from_confusion(confusion_matrix(y_true, y_pred, len(class_names)), class_names)


def class_weights_sqrt_inverse(labels: Sequence[int], num_classes: int) -> list[float]:
    """(N / (K * n_c)) ** 0.5 from the given (training) labels. Absent classes get 0."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    n = counts.sum()
    weights = np.zeros(num_classes)
    nonzero = counts > 0
    weights[nonzero] = np.sqrt(n / (num_classes * counts[nonzero]))
    return [float(w) for w in weights]

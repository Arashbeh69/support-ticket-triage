"""Shared multiclass evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss


def multiclass_brier(y_true_indices: np.ndarray, probabilities: np.ndarray) -> float:
    """Return the mean multiclass Brier score across observations."""
    one_hot = np.zeros_like(probabilities, dtype=float)
    one_hot[np.arange(len(y_true_indices)), y_true_indices] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def top_k_accuracy(y_true_indices: np.ndarray, probabilities: np.ndarray, k: int) -> float:
    """Return the share of true labels appearing among the k largest probabilities."""
    top = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(top == y_true_indices[:, None], axis=1)))


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    probability_labels: np.ndarray,
) -> dict[str, Any]:
    """Calculate the model-selection metrics without consulting test data."""
    label_to_index = {label: index for index, label in enumerate(probability_labels)}
    y_indices = np.array([label_to_index[label] for label in y_true], dtype=int)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "top_3_accuracy": top_k_accuracy(y_indices, probabilities, k=3),
        "log_loss": float(log_loss(y_true, probabilities, labels=probability_labels)),
        "multiclass_brier": multiclass_brier(y_indices, probabilities),
    }

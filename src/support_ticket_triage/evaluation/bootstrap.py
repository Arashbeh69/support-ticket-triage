"""Paired bootstrap uncertainty for model-comparison metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score


def paired_macro_f1_difference(
    y_true: np.ndarray,
    baseline_prediction: np.ndarray,
    transformer_prediction: np.ndarray,
    labels: np.ndarray,
    iterations: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Bootstrap paired rows and return transformer-minus-baseline macro-F1 uncertainty."""
    if not (
        len(y_true) == len(baseline_prediction) == len(transformer_prediction) and iterations > 0
    ):
        raise ValueError("paired predictions must have equal non-zero lengths")
    rng = np.random.default_rng(seed)
    differences = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sample = rng.integers(0, len(y_true), size=len(y_true))
        baseline_score = f1_score(
            y_true[sample],
            baseline_prediction[sample],
            labels=labels,
            average="macro",
            zero_division=0,
        )
        transformer_score = f1_score(
            y_true[sample],
            transformer_prediction[sample],
            labels=labels,
            average="macro",
            zero_division=0,
        )
        differences[index] = transformer_score - baseline_score
    alpha = 1.0 - confidence_level
    point = f1_score(
        y_true,
        transformer_prediction,
        labels=labels,
        average="macro",
        zero_division=0,
    ) - f1_score(
        y_true,
        baseline_prediction,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    return {
        "contrast": "transformer minus baseline test macro F1",
        "point_difference": float(point),
        "confidence_level": confidence_level,
        "lower": float(np.quantile(differences, alpha / 2)),
        "upper": float(np.quantile(differences, 1 - alpha / 2)),
        "iterations": iterations,
        "seed": seed,
        "paired_resampling_unit": "official test row",
    }

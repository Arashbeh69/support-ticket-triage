"""Validation-only probability calibration and selective-classification policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar


def softmax(logits: np.ndarray) -> np.ndarray:
    """Apply a numerically stable row-wise softmax."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def temperature_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Convert logits to probabilities using a positive scalar temperature."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return softmax(logits / temperature)


def fit_temperature(logits: np.ndarray, y_true_indices: np.ndarray) -> float:
    """Fit one temperature by validation negative log likelihood."""
    if logits.ndim != 2 or len(logits) != len(y_true_indices):
        raise ValueError("logits and targets have incompatible shapes")

    def objective(log_temperature: float) -> float:
        probabilities = temperature_probabilities(logits, float(np.exp(log_temperature)))
        selected = np.clip(probabilities[np.arange(len(y_true_indices)), y_true_indices], 1e-12, 1)
        return float(-np.log(selected).mean())

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"temperature optimization failed: {result.message}")
    return float(np.exp(result.x))


def reliability_table(
    y_true_indices: np.ndarray,
    probabilities: np.ndarray,
    bins: int,
) -> list[dict[str, Any]]:
    """Return equal-width confidence-bin counts, accuracy, confidence, and gap."""
    if bins < 2:
        raise ValueError("at least two bins are required")
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == y_true_indices
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(confidence, edges[1:-1]), bins - 1)
    rows: list[dict[str, Any]] = []
    for index in range(bins):
        mask = assignments == index
        count = int(mask.sum())
        mean_confidence = float(confidence[mask].mean()) if count else None
        accuracy = float(correct[mask].mean()) if count else None
        rows.append(
            {
                "bin": index + 1,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "absolute_gap": (abs(float(mean_confidence) - float(accuracy)) if count else None),
            }
        )
    return rows


def expected_calibration_error(
    y_true_indices: np.ndarray,
    probabilities: np.ndarray,
    bins: int,
) -> float:
    """Return count-weighted absolute accuracy-confidence gap in equal-width bins."""
    table = reliability_table(y_true_indices, probabilities, bins)
    total = len(y_true_indices)
    return float(sum(row["count"] / total * row["absolute_gap"] for row in table if row["count"]))


@dataclass(frozen=True)
class ReviewThresholds:
    confidence: float
    margin: float
    coverage: float
    selective_error: float
    automated_rows: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "margin": self.margin,
            "coverage": self.coverage,
            "selective_error": self.selective_error,
            "automated_rows": self.automated_rows,
        }


def confidence_and_margin(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return top-one confidence and top-one minus top-two probability margin."""
    top_two = np.partition(probabilities, -2, axis=1)[:, -2:]
    top_two.sort(axis=1)
    return top_two[:, 1], top_two[:, 1] - top_two[:, 0]


def review_mask(
    probabilities: np.ndarray,
    predicted_labels: np.ndarray,
    confidence_threshold: float,
    margin_threshold: float,
    high_risk_intents: set[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Apply model-output review rules that do not require raw input text."""
    confidence, margin = confidence_and_margin(probabilities)
    reasons = {
        "low_confidence": confidence < confidence_threshold,
        "small_top_two_margin": margin < margin_threshold,
        "high_risk_intent": np.isin(predicted_labels, list(high_risk_intents)),
    }
    combined = np.logical_or.reduce(list(reasons.values()))
    return combined, reasons


def select_review_thresholds(
    y_true_indices: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    high_risk_intents: set[str],
    confidence_grid: list[float],
    margin_grid: list[float],
    maximum_selective_error: float,
) -> tuple[ReviewThresholds, list[dict[str, Any]]]:
    """Maximize validation coverage subject to an error-rate constraint."""
    prediction_indices = probabilities.argmax(axis=1)
    predicted_labels = labels[prediction_indices]
    candidates: list[ReviewThresholds] = []
    curve: list[dict[str, Any]] = []
    for confidence_threshold in confidence_grid:
        for margin_threshold in margin_grid:
            review, _ = review_mask(
                probabilities,
                predicted_labels,
                float(confidence_threshold),
                float(margin_threshold),
                high_risk_intents,
            )
            automated = ~review
            automated_rows = int(automated.sum())
            selective_error = (
                float((prediction_indices[automated] != y_true_indices[automated]).mean())
                if automated_rows
                else 1.0
            )
            candidate = ReviewThresholds(
                confidence=float(confidence_threshold),
                margin=float(margin_threshold),
                coverage=float(automated.mean()),
                selective_error=selective_error,
                automated_rows=automated_rows,
            )
            curve.append(candidate.as_dict())
            if automated_rows and selective_error <= maximum_selective_error:
                candidates.append(candidate)
    if not candidates:
        raise RuntimeError("no review threshold satisfies the validation error constraint")
    selected = sorted(
        candidates,
        key=lambda item: (-item.coverage, item.selective_error, item.confidence + item.margin),
    )[0]
    return selected, curve

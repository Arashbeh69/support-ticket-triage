"""Focused tests for calibration and review-threshold behavior."""

import numpy as np
import pytest

from support_ticket_triage.evaluation.calibration import (
    expected_calibration_error,
    fit_temperature,
    review_mask,
    select_review_thresholds,
    temperature_probabilities,
)


def test_temperature_probabilities_are_normalized() -> None:
    probabilities = temperature_probabilities(np.array([[2.0, 0.0], [0.0, 2.0]]), 2.0)
    assert probabilities.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert np.all(probabilities > 0)


def test_temperature_fit_improves_overconfident_nll() -> None:
    logits = np.array([[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [0.0, 8.0]])
    targets = np.array([0, 1, 1, 0])
    assert fit_temperature(logits, targets) > 1.0


def test_ece_is_zero_for_matching_bin_accuracy_and_confidence() -> None:
    probabilities = np.array([[0.75, 0.25], [0.75, 0.25], [0.25, 0.75], [0.25, 0.75]])
    targets = np.array([0, 0, 1, 0])
    assert expected_calibration_error(targets, probabilities, bins=4) == pytest.approx(0.0)


def test_review_mask_always_routes_high_risk_predictions_to_review() -> None:
    probabilities = np.array([[0.95, 0.05], [0.4, 0.6]])
    labels = np.array(["routine", "high_risk"])
    review, reasons = review_mask(probabilities, labels, 0.5, 0.2, {"high_risk"})
    assert review.tolist() == [False, True]
    assert reasons["high_risk_intent"].tolist() == [False, True]


def test_threshold_selection_obeys_error_constraint() -> None:
    probabilities = np.array([[0.95, 0.05], [0.9, 0.1], [0.55, 0.45], [0.51, 0.49]], dtype=float)
    targets = np.array([0, 0, 1, 1])
    labels = np.array(["a", "b"])
    selected, _ = select_review_thresholds(
        targets,
        probabilities,
        labels,
        set(),
        confidence_grid=[0.0, 0.8],
        margin_grid=[0.0],
        maximum_selective_error=0.05,
    )
    assert selected.confidence == 0.8
    assert selected.coverage == pytest.approx(0.5)

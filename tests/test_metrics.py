import numpy as np
import pytest

from support_ticket_triage.evaluation.metrics import (
    classification_metrics,
    multiclass_brier,
    top_k_accuracy,
)


def test_metrics_are_perfect_for_perfect_predictions() -> None:
    labels = np.array(["a", "b", "c"])
    truth = np.array(["a", "b", "c"])
    probabilities = np.eye(3)
    result = classification_metrics(truth, truth, probabilities, labels)
    assert result["macro_f1"] == 1.0
    assert result["weighted_f1"] == 1.0
    assert result["accuracy"] == 1.0
    assert result["top_3_accuracy"] == 1.0
    assert multiclass_brier(np.array([0, 1, 2]), probabilities) == 0.0


def test_top_k_uses_probability_columns() -> None:
    probabilities = np.array([[0.2, 0.7, 0.1], [0.4, 0.35, 0.25]])
    assert top_k_accuracy(np.array([1, 2]), probabilities, k=1) == 0.5
    assert top_k_accuracy(np.array([1, 2]), probabilities, k=2) == 0.5


def test_log_loss_uses_declared_probability_column_order() -> None:
    labels = np.array(["z_label", "a_label"])
    truth = np.array(["z_label", "a_label"])
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8]])
    result = classification_metrics(truth, truth, probabilities, labels)
    assert result["log_loss"] == pytest.approx(-(np.log(0.9) + np.log(0.8)) / 2)

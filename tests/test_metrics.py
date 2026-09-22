import numpy as np

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

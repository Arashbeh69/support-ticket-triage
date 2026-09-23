"""Tests for paired model-comparison uncertainty."""

import numpy as np

from support_ticket_triage.evaluation.bootstrap import paired_macro_f1_difference


def test_paired_bootstrap_reports_positive_perfect_model_gain() -> None:
    truth = np.array([0, 0, 1, 1, 2, 2])
    baseline = np.array([0, 1, 1, 2, 2, 0])
    transformer = truth.copy()
    result = paired_macro_f1_difference(
        truth, baseline, transformer, np.array([0, 1, 2]), 100, 7, 0.95
    )
    assert result["point_difference"] > 0
    assert result["iterations"] == 100

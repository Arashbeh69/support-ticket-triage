"""Prediction-contract and input-validation tests using authored synthetic values."""

import numpy as np
import pytest

from support_ticket_triage.inference import (
    InputValidationError,
    decision_from_probabilities,
    validate_and_normalize,
)

POLICY = {
    "maximum_characters": 20,
    "maximum_utf8_bytes": 40,
    "minimum_non_whitespace_characters": 2,
}


def test_input_contract_rejects_control_characters_and_excess_length() -> None:
    with pytest.raises(InputValidationError, match="control"):
        validate_and_normalize("card\ncharge", POLICY)
    with pytest.raises(InputValidationError, match="character"):
        validate_and_normalize("x" * 21, POLICY)


def test_input_contract_normalizes_safe_whitespace() -> None:
    assert validate_and_normalize("  card   charge  ", POLICY) == "card charge"


def test_decision_returns_top_three_and_nonexclusive_review_reasons() -> None:
    labels = np.array(["routine", "high_risk", "other"])
    aliases = {label: label.title() for label in labels}
    routing = {label: "group" for label in labels}
    policy = {
        "confidence_threshold": 0.8,
        "top_one_top_two_margin_threshold": 0.2,
        "high_risk_intents": ["high_risk"],
    }
    result = decision_from_probabilities(
        np.array([0.1, 0.5, 0.4]),
        labels,
        aliases,
        routing,
        policy,
        privacy_detected=True,
        domain_mismatch=True,
        model_name="synthetic",
        model_version="test",
    )
    assert result.predicted_intent == "high_risk"
    assert len(result.top_three) == 3
    assert result.review_required
    assert set(result.review_reasons) == {
        "low_confidence",
        "small_top_two_margin",
        "privacy_or_secret_detection",
        "obvious_domain_mismatch",
        "high_risk_intent",
    }

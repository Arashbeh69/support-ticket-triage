"""Prediction-contract and input-validation tests using authored synthetic values."""

import json
from pathlib import Path

import numpy as np
import pytest

from support_ticket_triage.inference import (
    InferenceEngine,
    InputValidationError,
    canonical_review_policy_version,
    contains_domain_keyword,
    decision_from_probabilities,
    validate_and_normalize,
)

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_POLICY_CASES = ROOT / "tests" / "fixtures" / "synthetic_domain_policy_cases.json"
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
        no_domain_keyword_detected=True,
        model_name="synthetic",
        model_version="test",
        review_policy_version="policy-test",
    )
    assert result.predicted_intent == "high_risk"
    assert len(result.top_three) == 3
    assert result.review_required
    assert result.review_policy_version == "policy-test"
    assert set(result.review_reasons) == {
        "low_confidence",
        "small_top_two_margin",
        "privacy_or_secret_detection",
        "no_domain_keyword_detected",
        "high_risk_intent",
    }


def _synthetic_engine() -> InferenceEngine:
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.input_policy = json.loads(
        (ROOT / "configs" / "review_policy.json").read_text(encoding="utf-8")
    )
    engine.labels = np.array(["passcode_forgotten", "routine", "other"])
    engine.aliases = {label: label.replace("_", " ").title() for label in engine.labels}
    engine.routing = {label: "synthetic" for label in engine.labels}
    engine.review_policy = {
        "confidence_threshold": 0.0,
        "top_one_top_two_margin_threshold": 0.4,
        "high_risk_intents": [],
    }
    engine.model_name = "synthetic"
    engine.model_version = "policy-regression"
    engine.review_policy_version = "synthetic-policy"
    engine._probabilities = lambda _: np.array([0.9999, 0.00006, 0.00004])
    return engine


def test_synthetic_domain_policy_cases_are_regression_checks() -> None:
    fixture = json.loads(SYNTHETIC_POLICY_CASES.read_text(encoding="utf-8"))
    assert fixture["classification"] == "synthetic regression checks; not production validation"
    terms = json.loads(
        (ROOT / "configs" / "review_policy.json").read_text(encoding="utf-8")
    )["domain_terms"]

    for case in fixture["cases"]:
        assert contains_domain_keyword(case["text"], terms) is case["domain_keyword_expected"]


def test_passcode_inputs_do_not_receive_missing_keyword_reason() -> None:
    engine = _synthetic_engine()
    texts = (
        "I forgot the passcode for the app and cannot log in",
        "I forgot the passcode for my banking app and cannot log in",
    )

    for text in texts:
        result = engine.predict(text)
        assert result.predicted_intent == "passcode_forgotten"
        assert "no_domain_keyword_detected" not in result.review_reasons


def test_unrelated_high_confidence_input_keeps_keyword_review_rule() -> None:
    result = _synthetic_engine().predict("How do I bake sourdough bread?")

    assert result.calibrated_confidence > 0.99
    assert result.review_required
    assert result.review_reasons == ["no_domain_keyword_detected"]


def test_keyword_matching_uses_whole_terms_not_incidental_substrings() -> None:
    assert contains_domain_keyword("Please reset my PIN", ["pin"])
    assert not contains_domain_keyword("Tell me about spinning", ["pin"])


def test_review_policy_version_is_stable_across_json_key_order() -> None:
    first = {"domain_terms": ["bank", "passcode"], "threshold": 0.4}
    reordered = {"threshold": 0.4, "domain_terms": ["bank", "passcode"]}

    assert canonical_review_policy_version(first) == canonical_review_policy_version(reordered)

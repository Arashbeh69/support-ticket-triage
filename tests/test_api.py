"""FastAPI success, failure, metadata, and logging-privacy tests."""

import logging

from fastapi.testclient import TestClient

from support_ticket_triage.api import create_app
from support_ticket_triage.inference import Candidate, InputValidationError, Prediction


class FakeEngine:
    def metadata(self):
        return {
            "model_name": "synthetic",
            "model_version": "test",
            "review_policy_version": "policy-test",
        }

    def predict(self, text: str) -> Prediction:
        if "invalid" in text:
            raise InputValidationError(
                "input does not meet the prediction contract; manual review required"
            )
        candidates = [
            Candidate("intent_a", "Intent A", "group", 0.8),
            Candidate("intent_b", "Intent B", "group", 0.15),
            Candidate("intent_c", "Intent C", "group", 0.05),
        ]
        return Prediction(
            predicted_intent="intent_a",
            display_name="Intent A",
            routing_group="group",
            calibrated_confidence=0.8,
            top_three=candidates,
            review_required=False,
            review_reasons=[],
            model_name="synthetic",
            model_version="test",
            review_policy_version="policy-test",
        )


def test_api_contracts_and_logs_do_not_include_raw_text(caplog) -> None:
    authored_text = "A wholly invented card question for the API test"
    caplog.set_level(logging.INFO, logger="support_ticket_triage.api")
    with TestClient(create_app(FakeEngine())) as client:
        assert client.get("/health").json() == {"status": "ready"}
        assert client.get("/metadata").json()["model_name"] == "synthetic"
        response = client.post("/predict", json={"text": authored_text})
    assert response.status_code == 200
    assert response.json()["predicted_intent"] == "intent_a"
    assert response.json()["review_policy_version"] == "policy-test"
    assert len(response.json()["top_three"]) == 3
    assert authored_text not in caplog.text


def test_api_rejects_bad_payload_and_sanitizes_model_error() -> None:
    with TestClient(create_app(FakeEngine())) as client:
        extra = client.post("/predict", json={"text": "safe", "unexpected": "field"})
        invalid = client.post("/predict", json={"text": "invalid"})
    assert extra.status_code == 422
    assert invalid.status_code == 422
    assert "invalid" not in invalid.text.lower()
    assert "manual review required" in invalid.json()["detail"]

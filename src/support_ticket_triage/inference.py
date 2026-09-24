"""Reproducible local inference and portfolio review-policy application."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np

from support_ticket_triage.data.split import sha256_file
from support_ticket_triage.evaluation.calibration import (
    confidence_and_margin,
    temperature_probabilities,
)
from support_ticket_triage.text import detect_sensitive, normalize_text, redact_text


class InputValidationError(ValueError):
    """Input cannot be safely scored and must be handled outside automation."""


ReviewReason = Literal[
    "low_confidence",
    "small_top_two_margin",
    "privacy_or_secret_detection",
    "no_domain_keyword_detected",
    "high_risk_intent",
]


@dataclass(frozen=True)
class Candidate:
    intent: str
    display_name: str
    routing_group: str
    probability: float


@dataclass(frozen=True)
class Prediction:
    predicted_intent: str
    display_name: str
    routing_group: str
    calibrated_confidence: float
    top_three: list[Candidate]
    review_required: bool
    review_reasons: list[ReviewReason]
    model_name: str
    model_version: str
    review_policy_version: str

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["top_three"] = [asdict(candidate) for candidate in self.top_three]
        return result


def validate_and_normalize(text: str, policy: dict[str, Any]) -> str:
    """Apply character, byte, whitespace, and control-character limits."""
    if not isinstance(text, str):
        raise InputValidationError(
            "input does not meet the prediction contract; manual review required"
        )
    if len(text) > int(policy["maximum_characters"]):
        raise InputValidationError("input exceeds the character limit; manual review required")
    if len(text.encode("utf-8")) > int(policy["maximum_utf8_bytes"]):
        raise InputValidationError("input exceeds the byte limit; manual review required")
    if any(unicodedata.category(character) == "Cc" for character in text):
        raise InputValidationError("input contains control characters; manual review required")
    normalized = normalize_text(text)
    if len(normalized) < int(policy["minimum_non_whitespace_characters"]):
        raise InputValidationError("input is empty or too short; manual review required")
    return normalized


def _routing_lookup(groups: dict[str, list[str]]) -> dict[str, str]:
    lookup = {label: group for group, labels in groups.items() for label in labels}
    if len(lookup) != 77:
        raise RuntimeError("routing groups do not cover exactly 77 labels")
    return lookup


def contains_domain_keyword(text: str, domain_terms: list[str]) -> bool:
    """Return whether a configured whole keyword or phrase occurs in normalized text.

    This transparent lexical guardrail is deliberately separate from model confidence. It
    is not a trained out-of-distribution detector.
    """
    lowered = text.casefold()
    return any(
        re.search(rf"(?<!\w){re.escape(term.casefold())}(?!\w)", lowered) is not None
        for term in domain_terms
    )


def canonical_review_policy_version(policy: dict[str, Any]) -> str:
    """Return a cross-platform short hash of canonical review-policy JSON."""
    payload = json.dumps(
        policy,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def decision_from_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
    aliases: dict[str, str],
    routing: dict[str, str],
    policy: dict[str, Any],
    privacy_detected: bool,
    no_domain_keyword_detected: bool,
    model_name: str,
    model_version: str,
    review_policy_version: str,
) -> Prediction:
    """Build the prediction contract and nonexclusive review reasons."""
    if probabilities.shape != (len(labels),):
        raise ValueError("probabilities do not match the label contract")
    ordering = np.argsort(probabilities)[::-1]
    predicted = str(labels[ordering[0]])
    confidence, margin = confidence_and_margin(probabilities[None, :])
    reasons: list[ReviewReason] = []
    if float(confidence[0]) < float(policy["confidence_threshold"]):
        reasons.append("low_confidence")
    if float(margin[0]) < float(policy["top_one_top_two_margin_threshold"]):
        reasons.append("small_top_two_margin")
    if privacy_detected:
        reasons.append("privacy_or_secret_detection")
    if no_domain_keyword_detected:
        reasons.append("no_domain_keyword_detected")
    if predicted in set(policy["high_risk_intents"]):
        reasons.append("high_risk_intent")
    top_three = [
        Candidate(
            intent=str(labels[index]),
            display_name=aliases[str(labels[index])],
            routing_group=routing[str(labels[index])],
            probability=float(probabilities[index]),
        )
        for index in ordering[:3]
    ]
    return Prediction(
        predicted_intent=predicted,
        display_name=aliases[predicted],
        routing_group=routing[predicted],
        calibrated_confidence=float(confidence[0]),
        top_three=top_three,
        review_required=bool(reasons),
        review_reasons=reasons,
        model_name=model_name,
        model_version=model_version,
        review_policy_version=review_policy_version,
    )


class InferenceEngine:
    """Load the frozen champion once and score text without logging it."""

    def __init__(self, project_root: Path, private_root: Path) -> None:
        self.project_root = project_root
        self.private_root = private_root
        self.labels = np.asarray(
            json.loads((project_root / "configs" / "labels.json").read_text(encoding="utf-8"))
        )
        alias_overrides = json.loads(
            (project_root / "configs" / "display_aliases.json").read_text(encoding="utf-8")
        )
        self.aliases = {
            str(label): alias_overrides.get(str(label), str(label).replace("_", " ").title())
            for label in self.labels
        }
        self.routing = _routing_lookup(
            json.loads(
                (project_root / "configs" / "routing_groups.json").read_text(encoding="utf-8")
            )
        )
        review_policy_path = project_root / "configs" / "review_policy.json"
        self.input_policy = json.loads(review_policy_path.read_text(encoding="utf-8"))
        self.evaluation = json.loads(
            (project_root / "configs" / "evaluation.json").read_text(encoding="utf-8")
        )
        self.review_policy = self.evaluation["review_policy"]
        self.model_name = str(self.evaluation["champion"])
        self.model_version = sha256_file(project_root / "configs" / "evaluation.json")[:16]
        self.review_policy_version = canonical_review_policy_version(self.input_policy)
        self._load_model()

    @classmethod
    def from_environment(cls) -> InferenceEngine:
        project_root = Path(__file__).resolve().parents[2]
        configured = os.environ.get("SUPPORT_TRIAGE_PRIVATE_ROOT")
        private_root = (
            Path(configured)
            if configured
            else project_root.with_name(f"{project_root.name}-private")
        )
        return cls(project_root, private_root)

    def _load_model(self) -> None:
        if self.model_name == "baseline":
            self.model = joblib.load(
                self.private_root / "models" / "baseline" / "tfidf_logistic.joblib"
            )
            classes = self.model.named_steps["classifier"].classes_
            self.reorder = np.array(
                [int(np.where(classes == label)[0][0]) for label in self.labels], dtype=int
            )
            return
        if self.model_name != "transformer":
            raise RuntimeError(f"unsupported frozen champion: {self.model_name}")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = self.private_root / "models" / "transformer"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True
        ).to("cpu")
        self.model.eval()
        self.torch = torch
        self.transformer_config = json.loads(
            (self.project_root / "configs" / "transformer.json").read_text(encoding="utf-8")
        )

    def _probabilities(self, text: str) -> np.ndarray:
        temperature = float(self.evaluation["calibration"][f"{self.model_name}_temperature"])
        if self.model_name == "baseline":
            raw = self.model.predict_proba([text])[0][self.reorder]
            return temperature_probabilities(
                np.log(np.clip(raw, 1e-12, 1.0))[None, :], temperature
            )[0]
        encoded = self.tokenizer(
            [text],
            padding="max_length",
            truncation=True,
            max_length=int(self.transformer_config["max_length"]),
            return_tensors="pt",
        )
        with self.torch.inference_mode():
            logits = self.model(**encoded).logits.numpy()
        return temperature_probabilities(logits, temperature)[0]

    def predict(self, text: str) -> Prediction:
        """Validate, redact, predict, and attach portfolio review reasons."""
        normalized = validate_and_normalize(text, self.input_policy)
        detections = detect_sensitive(normalized)
        model_text = redact_text(normalized, detections)
        no_domain_keyword_detected = not contains_domain_keyword(
            normalized, self.input_policy["domain_terms"]
        )
        probabilities = self._probabilities(model_text)
        return decision_from_probabilities(
            probabilities,
            self.labels,
            self.aliases,
            self.routing,
            self.review_policy,
            bool(detections),
            no_domain_keyword_detected,
            self.model_name,
            self.model_version,
            self.review_policy_version,
        )

    def metadata(self) -> dict[str, Any]:
        """Return only public frozen metadata and operational limitations."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "review_policy_version": self.review_policy_version,
            "source_labels": len(self.labels),
            "language": "English",
            "domain": "banking support intents",
            "review_policy": "portfolio design proposal; not production validated",
            "limitations": [
                "no explicit unknown-intent class",
                "confidence is not correctness",
                "domain review is a transparent keyword guardrail, not a trained OOD detector",
                "abstention is not proven open-set detection",
                "not financial, security, fraud, or identity-verification advice",
            ],
        }

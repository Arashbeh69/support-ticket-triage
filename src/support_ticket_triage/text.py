"""Deterministic normalization, privacy detection, and redaction."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Detection:
    """One auditable privacy or sensitive-content detection."""

    rule: str
    reason: str
    start: int
    end: int
    severity: str
    match_sha256: str

    def as_dict(self) -> dict[str, str | int]:
        return asdict(self)


@dataclass(frozen=True)
class Rule:
    name: str
    reason: str
    pattern: re.Pattern[str]
    severity: str
    replacement: str


RULES = (
    Rule(
        "email",
        "Email address candidate",
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
        "high",
        "<EMAIL>",
    ),
    Rule(
        "url",
        "Web URL candidate",
        re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE),
        "medium",
        "<URL>",
    ),
    Rule(
        "social_handle",
        "Social-media handle candidate",
        re.compile(r"(?<!\w)@[A-Za-z0-9_]{2,32}\b"),
        "medium",
        "<SOCIAL_HANDLE>",
    ),
    Rule(
        "iban",
        "IBAN-like identifier",
        re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}\b"),
        "high",
        "<IBAN>",
    ),
    Rule(
        "credential_value",
        "Credential keyword followed by a value",
        re.compile(
            r"\b(?:password|passcode|pin|otp|token|secret|cvv)\s*(?:is|was|:|=)\s*[A-Za-z0-9!@#$%^&*_-]{3,}\b",
            re.IGNORECASE,
        ),
        "high",
        "<CREDENTIAL>",
    ),
    Rule(
        "uk_postcode",
        "UK postcode candidate",
        re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.IGNORECASE),
        "medium",
        "<POSTCODE>",
    ),
    Rule(
        "street_address",
        "Street-address candidate",
        re.compile(
            r"\b\d{1,5}\s+[A-Za-z][A-Za-z .'-]{2,30}\s(?:street|st|road|rd|avenue|ave|lane|ln)\b",
            re.IGNORECASE,
        ),
        "medium",
        "<ADDRESS>",
    ),
    Rule(
        "name_or_signature",
        "Name or signature phrase candidate",
        re.compile(
            r"\b(?:my name is|this is|regards|sincerely)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"
        ),
        "medium",
        "<NAME>",
    ),
    Rule(
        "long_digit_sequence",
        "Long numeric identifier candidate",
        re.compile(r"\b\d{6,}\b"),
        "medium",
        "<LONG_NUMBER>",
    ),
    Rule(
        "phone_candidate",
        "Phone-number candidate requiring review",
        re.compile(r"(?<!\w)(?:\+?\d[\d .()/-]{6,}\d)(?!\w)"),
        "medium",
        "<PHONE>",
    ),
)


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving wording and case."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def audit_key(text: str) -> str:
    """Return the conservative case-folded representation used for duplicate audits."""
    normalized = normalize_text(text).casefold()
    return re.sub(r"[^\w\s]", " ", normalized).strip()


def template_key(text: str) -> str:
    """Generalize obvious variable slots for template-frequency analysis."""
    value = audit_key(text)
    value = re.sub(r"\b\d+(?:[.,]\d+)?\b", "<number>", value)
    value = re.sub(r"\b(?:gbp|eur|usd|pounds?|euros?|dollars?)\b", "<currency>", value)
    return " ".join(value.split())


def _luhn_valid(candidate: str) -> bool:
    digits = [int(char) for char in candidate if char.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def detect_sensitive(text: str) -> list[Detection]:
    """Detect candidates without assuming every number is personal information."""
    findings: list[Detection] = []
    occupied: set[tuple[int, int, str]] = set()
    for rule in RULES:
        for match in rule.pattern.finditer(text):
            key = (match.start(), match.end(), rule.name)
            if key in occupied:
                continue
            occupied.add(key)
            findings.append(
                Detection(
                    rule=rule.name,
                    reason=rule.reason,
                    start=match.start(),
                    end=match.end(),
                    severity=rule.severity,
                    match_sha256=hashlib.sha256(match.group(0).encode("utf-8")).hexdigest(),
                )
            )
    card_pattern = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
    for match in card_pattern.finditer(text):
        if _luhn_valid(match.group(0)):
            findings.append(
                Detection(
                    rule="card_number",
                    reason="Luhn-valid card-number candidate",
                    start=match.start(),
                    end=match.end(),
                    severity="high",
                    match_sha256=hashlib.sha256(match.group(0).encode("utf-8")).hexdigest(),
                )
            )
    return sorted(findings, key=lambda item: (item.start, item.end, item.rule))


def redact_text(text: str, detections: list[Detection]) -> str:
    """Replace detected spans with semantic placeholders, resolving overlap safely."""
    replacements = {rule.name: rule.replacement for rule in RULES}
    replacements["card_number"] = "<CARD_NUMBER>"
    selected: list[Detection] = []
    for item in sorted(detections, key=lambda found: (found.start, -(found.end - found.start))):
        if any(item.start < other.end and item.end > other.start for other in selected):
            continue
        selected.append(item)
    output = text
    for item in sorted(selected, key=lambda found: found.start, reverse=True):
        output = output[: item.start] + replacements[item.rule] + output[item.end :]
    return output

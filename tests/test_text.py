from support_ticket_triage.text import audit_key, detect_sensitive, normalize_text, redact_text


def test_normalization_is_deterministic() -> None:
    text = "  My\u00a0card   is not working.  "
    assert normalize_text(text) == "My card is not working."
    assert audit_key(text) == "my card is not working"


def test_email_is_detected_and_redacted() -> None:
    text = "Please contact me at safe.example@example.com"
    detections = detect_sensitive(text)
    assert [item.rule for item in detections] == ["email"]
    assert redact_text(text, detections) == "Please contact me at <EMAIL>"


def test_generic_pin_topic_is_not_treated_as_a_credential_value() -> None:
    assert detect_sensitive("I forgot my PIN") == []


def test_explicit_pin_value_is_quarantined() -> None:
    findings = detect_sensitive("My PIN is 1234")
    assert any(item.rule == "credential_value" and item.severity == "high" for item in findings)

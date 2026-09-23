"""Measure token lengths for the frozen development and validation data."""

import json
from pathlib import Path

from support_ticket_triage.models.transformer import token_length_audit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = PROJECT_ROOT.with_name(f"{PROJECT_ROOT.name}-private")


if __name__ == "__main__":
    print(json.dumps(token_length_audit(PROJECT_ROOT, PRIVATE_ROOT), indent=2, sort_keys=True))

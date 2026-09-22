"""Run privacy, duplicate, template, and leakage audits."""

import json
from pathlib import Path

from support_ticket_triage.data.audit import run_audit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = PROJECT_ROOT.with_name(f"{PROJECT_ROOT.name}-private")


if __name__ == "__main__":
    result = run_audit(PROJECT_ROOT, PRIVATE_ROOT)
    print(json.dumps(result, indent=2, sort_keys=True))

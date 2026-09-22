"""Build the duplicate-group-aware development/validation split."""

import json
from pathlib import Path

from support_ticket_triage.data.split import build_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = PROJECT_ROOT.with_name(f"{PROJECT_ROOT.name}-private")


if __name__ == "__main__":
    print(json.dumps(build_split(PROJECT_ROOT, PRIVATE_ROOT), indent=2, sort_keys=True))

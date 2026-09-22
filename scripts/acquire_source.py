"""Acquire the pinned BANKING77 files into the private sibling directory."""

from pathlib import Path

from support_ticket_triage.data.source import acquire_and_verify

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = PROJECT_ROOT.with_name(f"{PROJECT_ROOT.name}-private")


if __name__ == "__main__":
    result = acquire_and_verify(PROJECT_ROOT / "configs" / "source.json", PRIVATE_ROOT)
    print(
        f"Verified {result['splits']['train.csv']['rows']:,} train and "
        f"{result['splits']['test.csv']['rows']:,} test rows at {result['commit']}."
    )

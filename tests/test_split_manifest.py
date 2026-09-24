import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from support_ticket_triage.data.split import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def _tracked_bytes(relative_path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f":{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def test_split_manifest_is_hashed_and_group_isolated() -> None:
    manifest_path = ROOT / "manifests" / "split_manifest.csv"
    summary = json.loads((ROOT / "manifests" / "split_summary.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(manifest_path)
    assert sha256_file(manifest_path) == summary["manifest_sha256"]
    assert hashlib.sha256(_tracked_bytes("manifests/split_manifest.csv")).hexdigest() == (
        summary["manifest_sha256"]
    )
    development = frame.loc[frame["partition"] == "development"]
    validation = frame.loc[frame["partition"] == "validation"]
    assert development["category"].nunique() == 77
    assert validation["category"].nunique() == 77
    assert not set(development["duplicate_group"]).intersection(validation["duplicate_group"])
    assert frame["row_id"].is_unique

"""Cross-platform byte-level checks for immutable public evaluation evidence."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tracked_bytes(relative_path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f":{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _assert_worktree_and_tracked_hash(relative_path: str, expected: str) -> None:
    assert _sha256((ROOT / relative_path).read_bytes()) == expected
    assert _sha256(_tracked_bytes(relative_path)) == expected


def test_evaluation_config_matches_frozen_and_official_result_hashes() -> None:
    frozen_hash = _json("manifests/evaluation_freeze.json")["evaluation_config_sha256"]
    official_hash = _json("reports/official_test_results.json")["evaluation_config_sha256"]

    assert frozen_hash == official_hash
    _assert_worktree_and_tracked_hash("configs/evaluation.json", frozen_hash)


def test_recorded_preprocessing_hashes_match_tracked_bytes() -> None:
    for relative_path, expected in _json("configs/evaluation.json")[
        "preprocessing_sha256"
    ].items():
        _assert_worktree_and_tracked_hash(relative_path, expected)


def test_attributes_preserve_frozen_crlf_exceptions_and_lf_elsewhere() -> None:
    result = subprocess.run(
        [
            "git",
            "check-attr",
            "text",
            "eol",
            "diff",
            "--",
            "manifests/split_manifest.csv",
            "configs/evaluation.json",
            "configs/labels.json",
            "src/support_ticket_triage/text.py",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert "manifests/split_manifest.csv: text: unset" in result
    assert "configs/evaluation.json: text: unset" in result
    assert "manifests/split_manifest.csv: diff: unset" in result
    assert "configs/evaluation.json: diff: unset" in result
    assert "configs/labels.json: eol: lf" in result
    assert "src/support_ticket_triage/text.py: eol: lf" in result

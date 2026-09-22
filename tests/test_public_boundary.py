import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".pkl", ".joblib", ".safetensors", ".onnx"}
FORBIDDEN_PARTS = {"raw", "quarantine", "checkpoints", "runs", "logs", "secrets"}
TEXT_COLUMNS = {"text", "raw_text", "normalized_text", "model_text", "redacted_text"}


def _tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / line for line in result.stdout.splitlines() if line]


def test_no_private_artifact_types_or_directories_are_tracked() -> None:
    for path in _tracked_files():
        relative = path.relative_to(ROOT)
        assert path.suffix.lower() not in FORBIDDEN_SUFFIXES
        assert not FORBIDDEN_PARTS.intersection(part.casefold() for part in relative.parts)
        assert path.stat().st_size < 50 * 1024 * 1024


def test_public_csv_files_do_not_contain_text_columns() -> None:
    for path in _tracked_files():
        if path.suffix.lower() != ".csv":
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle))
        assert not TEXT_COLUMNS.intersection(column.casefold() for column in header)

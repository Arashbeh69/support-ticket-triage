"""Acquire and verify the pinned BANKING77 source without publishing raw text."""

from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_contract(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "support-ticket-triage-ml/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        destination.write_bytes(response.read())


def _inspect_csv(path: Path, expected_schema: list[str]) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_schema:
            raise ValueError(f"Unexpected schema for {path.name}: {reader.fieldnames}")
        counts: dict[str, int] = {}
        rows = 0
        empty_text = 0
        for row in reader:
            rows += 1
            label = row["category"]
            counts[label] = counts.get(label, 0) + 1
            empty_text += int(not row["text"].strip())
    return {"rows": rows, "empty_text": empty_text, "class_counts": counts}


def acquire_and_verify(contract_path: Path, private_root: Path) -> dict[str, Any]:
    """Download pinned source files privately and write a verified metadata record."""
    contract = _load_contract(contract_path)
    commit = contract["commit"]
    base = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{commit}"
    raw_dir = private_root / "data" / "raw" / "banking77"
    raw_dir.mkdir(parents=True, exist_ok=True)

    urls: dict[str, str] = {}
    for name, relative in contract["retrieval_files"].items():
        url = f"{base}/{relative}"
        urls[name] = url
        _download(url, raw_dir / name)

    categories = json.loads((raw_dir / "categories.json").read_text(encoding="utf-8"))
    if len(categories) != contract["expected_label_count"] or len(set(categories)) != len(categories):
        raise ValueError("Source categories are not 77 unique labels")

    split_audits = {
        name: _inspect_csv(raw_dir / name, contract["expected_schema"])
        for name in contract["expected_counts"]
    }
    for name, expected in contract["expected_counts"].items():
        if split_audits[name]["rows"] != expected:
            raise ValueError(f"Unexpected row count for {name}: {split_audits[name]['rows']}")
        unknown = set(split_audits[name]["class_counts"]) - set(categories)
        missing = set(categories) - set(split_audits[name]["class_counts"])
        if unknown or missing:
            raise ValueError(f"Label mismatch for {name}; unknown={unknown}, missing={missing}")

    license_text = (raw_dir / "LICENSE").read_text(encoding="utf-8")
    if "Attribution 4.0 International" not in license_text:
        raise ValueError("Pinned licence does not identify CC BY 4.0")

    metadata = {
        "dataset": contract["dataset"],
        "repository": contract["repository"],
        "commit": commit,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_urls": urls,
        "licence": contract["licence"],
        "schema": contract["expected_schema"],
        "files": {
            name: {"sha256": sha256_file(raw_dir / name), "bytes": (raw_dir / name).stat().st_size}
            for name in contract["retrieval_files"]
        },
        "splits": split_audits,
        "labels": categories,
    }
    metadata_path = private_root / "validation" / "source_audit.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


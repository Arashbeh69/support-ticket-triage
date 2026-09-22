"""Leakage-aware development and validation split construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def sha256_file(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _choose_validation_fold(
    frame: pd.DataFrame, folds: int, seed: int, labels: list[str]
) -> tuple[list[int], list[int]]:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    target_fraction = 1.0 / folds
    candidates: list[tuple[tuple[int, float, int], list[int], list[int]]] = []
    for development, validation in splitter.split(
        frame, y=frame["category"], groups=frame["duplicate_group"]
    ):
        validation_labels = set(frame.iloc[validation]["category"])
        development_labels = set(frame.iloc[development]["category"])
        missing = len(set(labels) - validation_labels) + len(set(labels) - development_labels)
        fraction_error = abs(len(validation) / len(frame) - target_fraction)
        support_spread = int(frame.iloc[validation].groupby("category").size().max()) - int(
            frame.iloc[validation].groupby("category").size().min()
        )
        candidates.append(
            ((missing, fraction_error, support_spread), development.tolist(), validation.tolist())
        )
    _, development, validation = min(candidates, key=lambda item: item[0])
    return development, validation


def build_split(project_root: Path, private_root: Path) -> dict[str, Any]:
    """Create and hash the split manifest without exposing source text publicly."""
    audit_config = json.loads((project_root / "configs" / "audit.json").read_text(encoding="utf-8"))
    labels = json.loads((project_root / "configs" / "labels.json").read_text(encoding="utf-8"))
    model_rows = pd.read_csv(
        private_root / "data" / "modelling" / "model_rows.csv", keep_default_na=False
    )
    audited = pd.read_csv(private_root / "manifests" / "audited_train_rows.csv")
    train_rows = model_rows.loc[model_rows["source_split"] == "train"].merge(
        audited, on=["row_id", "category"], how="inner", validate="one_to_one"
    )
    available = train_rows.loc[~train_rows["excluded_test_overlap"]].reset_index(drop=True)
    development_indices, validation_indices = _choose_validation_fold(
        available,
        folds=int(audit_config["validation_folds"]),
        seed=int(audit_config["split_seed"]),
        labels=labels,
    )
    available["partition"] = ""
    available.loc[development_indices, "partition"] = "development"
    available.loc[validation_indices, "partition"] = "validation"
    if (available["partition"] == "").any():
        raise RuntimeError("Not every eligible row was assigned")

    excluded = train_rows.loc[train_rows["excluded_test_overlap"]].copy()
    excluded["partition"] = "excluded_test_overlap"
    combined = pd.concat([available, excluded], ignore_index=True)
    public_columns = [
        "row_id",
        "source_index",
        "category",
        "duplicate_group",
        "partition",
    ]
    public_manifest = combined[public_columns].sort_values(["partition", "row_id"])
    manifest_dir = project_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "split_manifest.csv"
    public_manifest.to_csv(manifest_path, index=False)

    private_manifest = combined[
        public_columns + ["model_text", "redacted_text", "privacy_flag", "quarantine"]
    ].sort_values(["partition", "row_id"])
    private_path = private_root / "manifests" / "split_manifest_private.csv"
    private_manifest.to_csv(private_path, index=False)

    dev = available.loc[available["partition"] == "development"]
    val = available.loc[available["partition"] == "validation"]
    if set(dev["duplicate_group"]).intersection(val["duplicate_group"]):
        raise RuntimeError("Duplicate group crosses development and validation")
    if set(dev["category"]) != set(labels) or set(val["category"]) != set(labels):
        raise RuntimeError("All 77 labels must remain represented in both partitions")

    summary = {
        "seed": int(audit_config["split_seed"]),
        "method": "StratifiedGroupKFold candidate closest to 1/7 validation",
        "near_duplicate_threshold": audit_config["near_duplicate_threshold"],
        "counts": {
            "development": len(dev),
            "validation": len(val),
            "excluded_test_overlap": len(excluded),
            "official_test_held_out": int((model_rows["source_split"] == "test").sum()),
        },
        "class_support": {
            "development_min": int(dev.groupby("category").size().min()),
            "development_max": int(dev.groupby("category").size().max()),
            "validation_min": int(val.groupby("category").size().min()),
            "validation_max": int(val.groupby("category").size().max()),
        },
        "labels_in_development": dev["category"].nunique(),
        "labels_in_validation": val["category"].nunique(),
        "duplicate_groups_crossing_partitions": 0,
        "test_used_for_split_selection": False,
        "manifest_sha256": sha256_file(manifest_path),
        "private_manifest_sha256": sha256_file(private_path),
    }
    summary_path = manifest_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary

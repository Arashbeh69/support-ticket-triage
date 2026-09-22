"""Privacy, duplicate, template, and source-label audits for BANKING77."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from support_ticket_triage.text import (
    audit_key,
    detect_sensitive,
    normalize_text,
    redact_text,
    template_key,
)


class UnionFind:
    """Small deterministic disjoint-set implementation for duplicate groups."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


@dataclass(frozen=True)
class NearDuplicateResult:
    groups: list[int]
    sensitivity_counts: dict[str, int]
    candidate_pairs_at_threshold: int
    train_indices_overlapping_test: list[int]
    cross_split_pairs: int


def _stable_row_id(source_split: str, index: int, commit: str) -> str:
    material = f"{commit}:{source_split}:{index}".encode()
    return hashlib.sha256(material).hexdigest()[:20]


def load_source_rows(private_root: Path, commit: str) -> pd.DataFrame:
    """Load pinned raw files and attach stable, non-text-derived row IDs."""
    raw_dir = private_root / "data" / "raw" / "banking77"
    frames: list[pd.DataFrame] = []
    for source_split in ("train", "test"):
        frame = pd.read_csv(raw_dir / f"{source_split}.csv", keep_default_na=False)
        frame.insert(0, "source_index", np.arange(len(frame), dtype=int))
        frame.insert(0, "source_split", source_split)
        frame.insert(
            0,
            "row_id",
            [_stable_row_id(source_split, index, commit) for index in range(len(frame))],
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _near_duplicate_audit(
    train_texts: pd.Series,
    test_texts: pd.Series,
    threshold: float,
    neighbors: int,
    sensitivity: list[float],
) -> NearDuplicateResult:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=30_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_matrix = vectorizer.fit_transform(train_texts)
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
    model.fit(train_matrix)
    distances, indices = model.kneighbors(
        train_matrix, n_neighbors=min(neighbors, len(train_texts))
    )
    pairs: dict[tuple[int, int], float] = {}
    for left, (row_distances, row_indices) in enumerate(zip(distances, indices, strict=True)):
        for distance, right in zip(row_distances, row_indices, strict=True):
            if left == right:
                continue
            pair = (min(left, int(right)), max(left, int(right)))
            pairs[pair] = max(pairs.get(pair, 0.0), 1.0 - float(distance))
    sensitivity_counts = {
        f"{candidate:.2f}": sum(score >= candidate for score in pairs.values())
        for candidate in sensitivity
    }
    union_find = UnionFind(len(train_texts))
    selected_pairs = 0
    for (left, right), score in pairs.items():
        if score >= threshold:
            union_find.union(left, right)
            selected_pairs += 1
    roots = [union_find.find(index) for index in range(len(train_texts))]
    root_to_group = {root: group for group, root in enumerate(sorted(set(roots)))}
    groups = [root_to_group[root] for root in roots]

    test_matrix = vectorizer.transform(test_texts)
    cross_distances, cross_indices = model.kneighbors(
        test_matrix, n_neighbors=min(neighbors, len(train_texts))
    )
    overlapping_train: set[int] = set()
    cross_pairs = 0
    for row_distances, row_indices in zip(cross_distances, cross_indices, strict=True):
        for distance, train_index in zip(row_distances, row_indices, strict=True):
            if 1.0 - float(distance) >= threshold:
                overlapping_train.add(int(train_index))
                cross_pairs += 1
    return NearDuplicateResult(
        groups=groups,
        sensitivity_counts=sensitivity_counts,
        candidate_pairs_at_threshold=selected_pairs,
        train_indices_overlapping_test=sorted(overlapping_train),
        cross_split_pairs=cross_pairs,
    )


def _write_private_representations(frame: pd.DataFrame, private_root: Path) -> None:
    normalized_dir = private_root / "data" / "normalized"
    modelling_dir = private_root / "data" / "modelling"
    quarantine_dir = private_root / "data" / "quarantine"
    for directory in (normalized_dir, modelling_dir, quarantine_dir):
        directory.mkdir(parents=True, exist_ok=True)

    frame[
        [
            "row_id",
            "source_split",
            "source_index",
            "category",
            "text",
            "normalized_text",
            "audit_key",
        ]
    ].to_csv(normalized_dir / "audit_rows.csv", index=False)
    frame[
        [
            "row_id",
            "source_split",
            "category",
            "model_text",
            "redacted_text",
            "privacy_flag",
            "quarantine",
        ]
    ].to_csv(modelling_dir / "model_rows.csv", index=False)
    frame.loc[
        frame["quarantine"],
        ["row_id", "source_split", "source_index", "category", "text", "redacted_text"],
    ].to_csv(quarantine_dir / "quarantine_rows.csv", index=False)


def run_audit(project_root: Path, private_root: Path) -> dict[str, Any]:
    """Run the complete pre-modelling audit and write public aggregate evidence."""
    source_audit = json.loads(
        (private_root / "validation" / "source_audit.json").read_text(encoding="utf-8")
    )
    config = json.loads((project_root / "configs" / "audit.json").read_text(encoding="utf-8"))
    labels = json.loads((project_root / "configs" / "labels.json").read_text(encoding="utf-8"))
    routing_groups = json.loads(
        (project_root / "configs" / "routing_groups.json").read_text(encoding="utf-8")
    )
    source_categories = source_audit["labels"]
    if labels != source_categories:
        raise ValueError("Public source-label contract differs from the pinned source")
    routed = [label for values in routing_groups.values() for label in values]
    if len(routed) != 77 or len(set(routed)) != 77 or set(routed) != set(labels):
        raise ValueError("Routing groups must cover every source label exactly once")

    frame = load_source_rows(private_root, source_audit["commit"])
    frame["normalized_text"] = frame["text"].map(normalize_text)
    frame["model_text"] = frame["normalized_text"]
    frame["audit_key"] = frame["text"].map(audit_key)
    frame["template_key"] = frame["text"].map(template_key)

    detections_rows: list[dict[str, Any]] = []
    redacted: list[str] = []
    privacy_flags: list[bool] = []
    quarantine_flags: list[bool] = []
    for row in frame.itertuples(index=False):
        detections = detect_sensitive(row.normalized_text)
        redacted.append(redact_text(row.normalized_text, detections))
        privacy_flags.append(bool(detections))
        quarantine_flags.append(any(item.severity == "high" for item in detections))
        for item in detections:
            detections_rows.append(
                {
                    "row_id": row.row_id,
                    **item.as_dict(),
                    "disposition": "quarantine" if item.severity == "high" else "review",
                }
            )
    frame["redacted_text"] = redacted
    frame["privacy_flag"] = privacy_flags
    frame["quarantine"] = quarantine_flags

    privacy_dir = private_root / "privacy"
    privacy_dir.mkdir(parents=True, exist_ok=True)
    detections_frame = pd.DataFrame(detections_rows)
    detections_frame.to_csv(privacy_dir / "detections.csv", index=False)
    _write_private_representations(frame, private_root)

    train = frame.loc[frame["source_split"] == "train"].reset_index(drop=True)
    test = frame.loc[frame["source_split"] == "test"].reset_index(drop=True)
    exact_groups = train.groupby("audit_key", sort=False)
    exact_duplicate_groups = int(sum(len(group) > 1 for _, group in exact_groups))
    exact_duplicate_rows = int(sum(len(group) for _, group in exact_groups if len(group) > 1))
    conflicting_exact_groups = int(
        sum(group["category"].nunique() > 1 for _, group in exact_groups if len(group) > 1)
    )
    template_groups = train.groupby("template_key", sort=False)
    templated_groups = int(
        sum(len(group) > 1 and group["audit_key"].nunique() > 1 for _, group in template_groups)
    )

    near = _near_duplicate_audit(
        train["audit_key"],
        test["audit_key"],
        float(config["near_duplicate_threshold"]),
        int(config["near_duplicate_candidate_neighbors"]),
        [float(value) for value in config["near_duplicate_threshold_sensitivity"]],
    )
    train["duplicate_group"] = near.groups
    train["excluded_test_overlap"] = False
    if near.train_indices_overlapping_test:
        train.loc[near.train_indices_overlapping_test, "excluded_test_overlap"] = True

    class_distribution = (
        frame.groupby(["source_split", "category"], sort=True).size().rename("count").reset_index()
    )
    reports_dir = project_root / "reports"
    manifests_dir = project_root / "manifests"
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    class_distribution.to_csv(reports_dir / "class_distribution.csv", index=False)

    rule_counts = Counter(row["rule"] for row in detections_rows)
    disposition_counts = Counter(row["disposition"] for row in detections_rows)
    privacy_summary = {
        "rows_flagged": int(frame["privacy_flag"].sum()),
        "rows_quarantined": int(frame["quarantine"].sum()),
        "detections_by_rule": dict(sorted(rule_counts.items())),
        "detections_by_disposition": dict(sorted(disposition_counts.items())),
        "raw_flagged_examples_published": False,
    }
    summary = {
        "source_commit": source_audit["commit"],
        "rows": {"train": len(train), "test": len(test)},
        "labels": len(labels),
        "class_distribution": {
            "train_min": int(
                class_distribution.loc[class_distribution["source_split"] == "train", "count"].min()
            ),
            "train_max": int(
                class_distribution.loc[class_distribution["source_split"] == "train", "count"].max()
            ),
            "test_min": int(
                class_distribution.loc[class_distribution["source_split"] == "test", "count"].min()
            ),
            "test_max": int(
                class_distribution.loc[class_distribution["source_split"] == "test", "count"].max()
            ),
        },
        "privacy": privacy_summary,
        "duplicates": {
            "exact_duplicate_groups_train": exact_duplicate_groups,
            "rows_in_exact_duplicate_groups_train": exact_duplicate_rows,
            "conflicting_label_exact_groups_train": conflicting_exact_groups,
            "generalized_template_groups_train": templated_groups,
            "near_duplicate_threshold": config["near_duplicate_threshold"],
            "near_duplicate_candidate_pairs_train": near.candidate_pairs_at_threshold,
            "near_duplicate_threshold_sensitivity": near.sensitivity_counts,
            "cross_split_candidate_pairs": near.cross_split_pairs,
            "training_rows_excluded_for_test_overlap": len(near.train_indices_overlapping_test),
        },
        "test_labels_used_for_model_selection": False,
    }
    (reports_dir / "data_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    source_manifest = {
        key: value for key, value in source_audit.items() if key not in {"labels", "splits"}
    }
    source_manifest["split_counts"] = {
        name: audit["rows"] for name, audit in source_audit["splits"].items()
    }
    source_manifest["label_count"] = len(labels)
    (manifests_dir / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    private_manifest = train[
        ["row_id", "source_index", "category", "duplicate_group", "excluded_test_overlap"]
    ]
    private_manifest_dir = private_root / "manifests"
    private_manifest_dir.mkdir(parents=True, exist_ok=True)
    private_manifest.to_csv(private_manifest_dir / "audited_train_rows.csv", index=False)
    return summary

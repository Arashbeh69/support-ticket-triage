import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_source_labels_are_exact_and_unique() -> None:
    labels = json.loads((ROOT / "configs" / "labels.json").read_text(encoding="utf-8"))
    assert len(labels) == 77
    assert len(set(labels)) == 77
    assert "Refund_not_showing_up" in labels
    assert "reverted_card_payment?" in labels


def test_routing_groups_cover_every_label_once() -> None:
    labels = json.loads((ROOT / "configs" / "labels.json").read_text(encoding="utf-8"))
    groups = json.loads((ROOT / "configs" / "routing_groups.json").read_text(encoding="utf-8"))
    routed = [label for group in groups.values() for label in group]
    assert len(routed) == 77
    assert len(set(routed)) == 77
    assert set(routed) == set(labels)

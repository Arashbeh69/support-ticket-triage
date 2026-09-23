"""Contract tests for the pinned transformer and measured sequence length."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_transformer_revision_is_pinned_and_licensed() -> None:
    config = json.loads((ROOT / "configs" / "transformer.json").read_text(encoding="utf-8"))
    assert config["base_model"] == "distilbert/distilroberta-base"
    assert re.fullmatch(r"[0-9a-f]{40}", config["revision"])
    assert config["licence"] == "Apache-2.0"


def test_max_length_comes_from_selection_partitions_only() -> None:
    config = json.loads((ROOT / "configs" / "transformer.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "reports" / "token_length_audit.json").read_text(encoding="utf-8"))
    assert audit["official_test_used"] is False
    assert audit["rows"] == 8416 + 1404
    assert audit["percentiles"]["100"] <= config["max_length"]
    assert audit["counts_above"][str(config["max_length"])] == 0


def test_training_configuration_records_reproducibility_controls() -> None:
    config = json.loads((ROOT / "configs" / "transformer.json").read_text(encoding="utf-8"))
    assert config["random_seed"] == 20260922
    assert config["selection_metric"] == "macro_f1"
    assert config["per_device_batch_size"] * config["gradient_accumulation_steps"] == 32
    assert config["early_stopping_patience"] == 2

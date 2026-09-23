"""Refresh validation metrics from the saved transformer without retraining."""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from support_ticket_triage.evaluation.metrics import classification_metrics
from support_ticket_triage.models.transformer import (
    load_private_split,
    load_transformer_config,
    predict_logits,
    set_reproducible_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = PROJECT_ROOT.with_name(f"{PROJECT_ROOT.name}-private")


def main() -> None:
    """Recompute only validation predictions and replace the incorrect log loss."""
    config = load_transformer_config(PROJECT_ROOT)
    prior_path = PRIVATE_ROOT / "validation" / "transformer_validation.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    _, validation = load_private_split(PRIVATE_ROOT)
    labels = json.loads((PROJECT_ROOT / "configs" / "labels.json").read_text(encoding="utf-8"))

    set_reproducible_seed(int(config["random_seed"]))
    model_dir = PRIVATE_ROOT / "models" / "transformer"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logits = predict_logits(
        model,
        tokenizer,
        validation[config["representation"]].tolist(),
        int(config["max_length"]),
        int(config["per_device_batch_size"]) * 2,
        device,
    )
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    truth = validation["category"].to_numpy()
    prior["validation"] = classification_metrics(
        truth,
        np.asarray([labels[index] for index in predictions]),
        probabilities,
        np.asarray(labels),
    )
    prior["metric_integrity_correction"] = {
        "reason": (
            "Recomputed validation log loss directly from the declared probability-column "
            "order; the prior sklearn call reordered nonlexicographic labels."
        ),
        "retrained": False,
        "affected_metrics": ["log_loss"],
    }
    for destination in (
        prior_path,
        PROJECT_ROOT / "reports" / "transformer_validation.json",
    ):
        destination.write_text(json.dumps(prior, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(prior["validation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

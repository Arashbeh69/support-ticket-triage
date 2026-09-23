"""Freeze validation-selected calibration, review policy, and evaluation hashes."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from support_ticket_triage.data.split import sha256_file
from support_ticket_triage.evaluation.calibration import (
    expected_calibration_error,
    fit_temperature,
    reliability_table,
    select_review_thresholds,
    temperature_probabilities,
)
from support_ticket_triage.evaluation.metrics import classification_metrics
from support_ticket_triage.models.transformer import predict_logits, set_reproducible_seed

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT.with_name(f"{ROOT.name}-private")


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int,
) -> dict[str, float]:
    predictions = labels[probabilities.argmax(axis=1)]
    result = classification_metrics(true_labels, predictions, probabilities, labels)
    label_to_id = {label: index for index, label in enumerate(labels)}
    indices = np.array([label_to_id[label] for label in true_labels], dtype=int)
    result["ece"] = expected_calibration_error(indices, probabilities, bins)
    return result


def _save_plots(reliability: pd.DataFrame, selective: pd.DataFrame) -> None:
    figures = ROOT / "reports" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    calibrated = reliability.loc[reliability["stage"] == "calibrated"]
    plt.figure(figsize=(7, 5))
    for model_name, group in calibrated.groupby("model"):
        observed = group.dropna(subset=["accuracy"])
        plt.plot(observed["mean_confidence"], observed["accuracy"], marker="o", label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.xlabel("Mean calibrated confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Validation reliability (15 equal-width bins)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "validation_reliability.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plotted = selective.loc[selective["automated_rows"] > 0].sort_values("coverage")
    plt.scatter(plotted["coverage"], plotted["selective_error"], s=18, alpha=0.65)
    plt.xlabel("Automated coverage")
    plt.ylabel("Selective error")
    plt.title("Validation coverage versus selective risk")
    plt.tight_layout()
    plt.savefig(figures / "validation_selective_risk.png", dpi=160)
    plt.close()


def main() -> None:
    if (PRIVATE / "validation" / "official_test_evaluation.json").exists():
        raise RuntimeError("official test evidence already exists; the freeze may not be changed")
    transformer_config = _json(ROOT / "configs" / "transformer.json")
    review_config = _json(ROOT / "configs" / "review_policy.json")
    audit_config = _json(ROOT / "configs" / "audit.json")
    labels = np.asarray(_json(ROOT / "configs" / "labels.json"))
    label_to_id = {label: index for index, label in enumerate(labels)}
    split = pd.read_csv(PRIVATE / "manifests" / "split_manifest_private.csv")
    validation = split.loc[split["partition"] == "validation"].reset_index(drop=True)
    representation = transformer_config["representation"]
    true_labels = validation["category"].to_numpy()
    true_indices = np.array([label_to_id[label] for label in true_labels], dtype=int)
    bins = int(review_config["ece_equal_width_bins"])

    baseline = joblib.load(PRIVATE / "models" / "baseline" / "tfidf_logistic.joblib")
    baseline_raw = baseline.predict_proba(validation[representation])
    baseline_labels = baseline.named_steps["classifier"].classes_
    reorder = np.array([int(np.where(baseline_labels == label)[0][0]) for label in labels])
    baseline_raw = baseline_raw[:, reorder]
    baseline_logits = np.log(np.clip(baseline_raw, 1e-12, 1.0))

    set_reproducible_seed(int(transformer_config["random_seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer_dir = PRIVATE / "models" / "transformer"
    tokenizer = AutoTokenizer.from_pretrained(transformer_dir, local_files_only=True)
    transformer = AutoModelForSequenceClassification.from_pretrained(
        transformer_dir, local_files_only=True
    ).to(device)
    transformer_logits = predict_logits(
        transformer,
        tokenizer,
        validation[representation].tolist(),
        int(transformer_config["max_length"]),
        int(transformer_config["per_device_batch_size"]) * 2,
        device,
    )
    transformer_raw = temperature_probabilities(transformer_logits, 1.0)

    temperatures = {
        "baseline": fit_temperature(baseline_logits, true_indices),
        "transformer": fit_temperature(transformer_logits, true_indices),
    }
    probabilities = {
        "baseline": {
            "raw": baseline_raw,
            "calibrated": temperature_probabilities(baseline_logits, temperatures["baseline"]),
        },
        "transformer": {
            "raw": transformer_raw,
            "calibrated": temperature_probabilities(
                transformer_logits, temperatures["transformer"]
            ),
        },
    }
    comparison: dict[str, dict[str, object]] = {}
    reliability_rows: list[dict[str, object]] = []
    for model_name, stages in probabilities.items():
        comparison[model_name] = {
            "temperature": temperatures[model_name],
            "raw": _metrics(true_labels, stages["raw"], labels, bins),
            "calibrated": _metrics(true_labels, stages["calibrated"], labels, bins),
        }
        for stage_name, model_probabilities in stages.items():
            for row in reliability_table(true_indices, model_probabilities, bins):
                reliability_rows.append({"model": model_name, "stage": stage_name, **row})

    baseline_metrics = comparison["baseline"]["raw"]
    transformer_metrics = comparison["transformer"]["raw"]
    macro_gain = transformer_metrics["macro_f1"] - baseline_metrics["macro_f1"]
    weighted_regression = baseline_metrics["weighted_f1"] - transformer_metrics["weighted_f1"]
    if macro_gain >= float(
        review_config["champion_minimum_macro_f1_gain"]
    ) and weighted_regression <= float(review_config["champion_maximum_weighted_f1_regression"]):
        champion = "transformer"
        selection_reason = "met prespecified validation macro-F1 gain and weighted-F1 guardrail"
    else:
        champion = "baseline"
        selection_reason = "transformer did not clear the prespecified operational gain rule"

    champion_probabilities = probabilities[champion]["calibrated"]
    thresholds, curve = select_review_thresholds(
        true_indices,
        champion_probabilities,
        labels,
        set(audit_config["high_risk_intents"]),
        [float(value) for value in review_config["confidence_grid"]],
        [float(value) for value in review_config["margin_grid"]],
        float(review_config["maximum_selective_error"]),
    )
    comparison["selection"] = {
        "champion": champion,
        "reason": selection_reason,
        "transformer_macro_f1_gain": macro_gain,
        "transformer_weighted_f1_regression": weighted_regression,
        "official_test_used": False,
    }
    comparison["review_thresholds"] = thresholds.as_dict()

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "model_validation_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    reliability = pd.DataFrame(reliability_rows)
    reliability.to_csv(reports / "reliability_validation.csv", index=False)
    selective = pd.DataFrame(curve)
    selective.to_csv(reports / "selective_risk_validation.csv", index=False)
    _save_plots(reliability, selective)

    private_predictions = pd.DataFrame(
        {
            "row_id": validation["row_id"],
            "true_label": true_labels,
            "baseline_prediction": labels[baseline_raw.argmax(axis=1)],
            "transformer_prediction": labels[transformer_raw.argmax(axis=1)],
            "baseline_confidence_calibrated": probabilities["baseline"]["calibrated"].max(axis=1),
            "transformer_confidence_calibrated": probabilities["transformer"]["calibrated"].max(
                axis=1
            ),
        }
    )
    private_predictions.to_csv(PRIVATE / "validation" / "validation_predictions.csv", index=False)

    baseline_manifest = _json(ROOT / "manifests" / "baseline_artifact.json")
    transformer_manifest = _json(ROOT / "manifests" / "transformer_artifact.json")
    preprocessing_paths = [
        ROOT / "src" / "support_ticket_triage" / "text.py",
        ROOT / "src" / "support_ticket_triage" / "data" / "audit.py",
        ROOT / "src" / "support_ticket_triage" / "data" / "split.py",
        ROOT / "configs" / "audit.json",
        ROOT / "configs" / "labels.json",
    ]
    evaluation = {
        "version": "1.0.0",
        "official_test_used_for_freeze": False,
        "labels_sha256": sha256_file(ROOT / "configs" / "labels.json"),
        "review_policy_config_sha256": sha256_file(ROOT / "configs" / "review_policy.json"),
        "split_manifest_sha256": sha256_file(ROOT / "manifests" / "split_manifest.csv"),
        "private_split_manifest_sha256": sha256_file(
            PRIVATE / "manifests" / "split_manifest_private.csv"
        ),
        "preprocessing_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
            for path in preprocessing_paths
        },
        "models": {
            "baseline": baseline_manifest,
            "transformer": transformer_manifest,
        },
        "calibration": {
            "method": "scalar temperature scaling",
            "selection_partition": "validation",
            "baseline_temperature": temperatures["baseline"],
            "transformer_temperature": temperatures["transformer"],
            "ece_definition": f"top-label ECE with {bins} equal-width confidence bins",
        },
        "champion": champion,
        "champion_rule": {
            "minimum_transformer_macro_f1_gain": review_config["champion_minimum_macro_f1_gain"],
            "maximum_transformer_weighted_f1_regression": review_config[
                "champion_maximum_weighted_f1_regression"
            ],
            "result": selection_reason,
        },
        "review_policy": {
            "objective": "maximize validation automated coverage subject to <=5% selective error",
            "confidence_threshold": thresholds.confidence,
            "top_one_top_two_margin_threshold": thresholds.margin,
            "always_review": [
                "privacy_or_secret_detection",
                "malformed_input",
                "obvious_domain_mismatch",
                "configured_high_risk_intent",
            ],
            "high_risk_intents": audit_config["high_risk_intents"],
            "portfolio_policy_not_production_validated": True,
        },
        "test_protocol": {
            "paired_bootstrap_iterations": 3000,
            "paired_bootstrap_seed": 20260923,
            "confidence_level": 0.95,
            "cpu_latency_warmup": 20,
            "cpu_latency_repetitions": 200,
        },
    }
    evaluation_path = ROOT / "configs" / "evaluation.json"
    evaluation_path.write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    freeze_manifest = {
        "evaluation_config_sha256": sha256_file(evaluation_path),
        "official_test_used": False,
        "frozen": True,
    }
    (ROOT / "manifests" / "evaluation_freeze.json").write_text(
        json.dumps(freeze_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (PRIVATE / "validation" / "evaluation_freeze.json").write_text(
        json.dumps({**freeze_manifest, "evaluation": evaluation}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"comparison": comparison, "freeze": freeze_manifest}, indent=2))


if __name__ == "__main__":
    main()

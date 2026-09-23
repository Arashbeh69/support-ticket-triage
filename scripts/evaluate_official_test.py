"""Perform the single frozen official-test evaluation and text-free reporting."""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from support_ticket_triage.data.split import sha256_file
from support_ticket_triage.evaluation.bootstrap import paired_macro_f1_difference
from support_ticket_triage.evaluation.calibration import (
    confidence_and_margin,
    expected_calibration_error,
    reliability_table,
    review_mask,
    temperature_probabilities,
)
from support_ticket_triage.evaluation.metrics import classification_metrics
from support_ticket_triage.models.transformer import predict_logits, set_reproducible_seed

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT.with_name(f"{ROOT.name}-private")


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_bundle(
    truth: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int,
) -> dict[str, float]:
    predictions = labels[probabilities.argmax(axis=1)]
    metrics = classification_metrics(truth, predictions, probabilities, labels)
    label_to_id = {label: index for index, label in enumerate(labels)}
    truth_ids = np.array([label_to_id[label] for label in truth], dtype=int)
    metrics["ece"] = expected_calibration_error(truth_ids, probabilities, bins)
    return metrics


def _write_confusion(
    model_name: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    labels: np.ndarray,
) -> None:
    raw = confusion_matrix(truth, prediction, labels=labels)
    normalized = confusion_matrix(truth, prediction, labels=labels, normalize="true")
    for name, matrix in (("raw", raw), ("normalized", normalized)):
        table = pd.DataFrame(matrix, index=labels, columns=labels)
        table.index.name = "true_label"
        table.to_csv(ROOT / "reports" / f"{model_name}_confusion_{name}.csv")
        plt.figure(figsize=(12, 10))
        plt.imshow(matrix, aspect="auto", cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted class index")
        plt.ylabel("True class index")
        plt.title(f"{model_name.title()} official-test confusion matrix ({name})")
        plt.tight_layout()
        plt.savefig(ROOT / "reports" / "figures" / f"{model_name}_confusion_{name}.png", dpi=160)
        plt.close()


def _latency_summary(samples_ms: list[float]) -> dict[str, float | int]:
    return {
        "repetitions": len(samples_ms),
        "p50_ms": float(np.percentile(samples_ms, 50)),
        "p95_ms": float(np.percentile(samples_ms, 95)),
        "mean_ms": float(statistics.fmean(samples_ms)),
    }


def _measure_baseline_latency(model, text: str, warmup: int, repetitions: int) -> dict[str, Any]:
    for _ in range(warmup):
        model.predict_proba([text])
    samples: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        model.predict_proba([text])
        samples.append((time.perf_counter() - started) * 1000)
    return _latency_summary(samples)


@torch.no_grad()
def _measure_transformer_latency(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    max_length: int,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    encoded = tokenizer(
        [text], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    model.to(device).eval()
    for _ in range(warmup):
        model(input_ids=input_ids, attention_mask=attention_mask)
    if device.type == "cuda":
        torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        model(input_ids=input_ids, attention_mask=attention_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    return _latency_summary(samples)


def _plot_reliability(rows: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    for model_name, group in rows.groupby("model"):
        observed = group.dropna(subset=["accuracy"])
        plt.plot(observed["mean_confidence"], observed["accuracy"], marker="o", label=model_name)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("Mean calibrated confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Official-test reliability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "reports" / "figures" / "test_reliability.png", dpi=160)
    plt.close()


def main() -> None:
    freeze_manifest = _json(ROOT / "manifests" / "evaluation_freeze.json")
    evaluation_path = ROOT / "configs" / "evaluation.json"
    if sha256_file(evaluation_path) != freeze_manifest["evaluation_config_sha256"]:
        raise RuntimeError("evaluation configuration differs from the frozen hash")
    state_path = PRIVATE / "validation" / "official_test_access.json"
    if state_path.exists():
        raise RuntimeError("official test evaluation has already started; do not rerun")
    state_path.write_text(
        json.dumps(
            {
                "completed": False,
                "evaluation_config_sha256": freeze_manifest["evaluation_config_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    evaluation = _json(evaluation_path)
    transformer_config = _json(ROOT / "configs" / "transformer.json")
    review_config = _json(ROOT / "configs" / "review_policy.json")
    if (
        sha256_file(ROOT / "configs" / "review_policy.json")
        != evaluation["review_policy_config_sha256"]
    ):
        raise RuntimeError("review-policy configuration differs from the frozen hash")
    labels = np.asarray(_json(ROOT / "configs" / "labels.json"))
    label_to_id = {label: index for index, label in enumerate(labels)}
    rows = pd.read_csv(PRIVATE / "data" / "modelling" / "model_rows.csv")
    test = rows.loc[rows["source_split"] == "test"].reset_index(drop=True)
    if len(test) != 3080 or test["category"].nunique() != 77:
        raise RuntimeError("official test count or label coverage changed")
    representation = transformer_config["representation"]
    truth = test["category"].to_numpy()
    truth_ids = np.array([label_to_id[label] for label in truth], dtype=int)
    bins = int(review_config["ece_equal_width_bins"])
    process = psutil.Process()
    rss_before_models = process.memory_info().rss

    baseline = joblib.load(PRIVATE / "models" / "baseline" / "tfidf_logistic.joblib")
    rss_after_baseline = process.memory_info().rss
    baseline_raw = baseline.predict_proba(test[representation])
    baseline_labels = baseline.named_steps["classifier"].classes_
    reorder = np.array([int(np.where(baseline_labels == label)[0][0]) for label in labels])
    baseline_raw = baseline_raw[:, reorder]
    baseline_logits = np.log(np.clip(baseline_raw, 1e-12, 1.0))
    baseline_calibrated = temperature_probabilities(
        baseline_logits, float(evaluation["calibration"]["baseline_temperature"])
    )

    set_reproducible_seed(int(transformer_config["random_seed"]))
    transformer_dir = PRIVATE / "models" / "transformer"
    tokenizer = AutoTokenizer.from_pretrained(transformer_dir, local_files_only=True)
    transformer = AutoModelForSequenceClassification.from_pretrained(
        transformer_dir, local_files_only=True
    )
    rss_after_transformer_cpu = process.memory_info().rss
    prediction_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer.to(prediction_device)
    transformer_logits = predict_logits(
        transformer,
        tokenizer,
        test[representation].tolist(),
        int(transformer_config["max_length"]),
        int(transformer_config["per_device_batch_size"]) * 2,
        prediction_device,
    )
    transformer_raw = temperature_probabilities(transformer_logits, 1.0)
    transformer_calibrated = temperature_probabilities(
        transformer_logits, float(evaluation["calibration"]["transformer_temperature"])
    )

    model_probabilities = {
        "baseline": (baseline_raw, baseline_calibrated),
        "transformer": (transformer_raw, transformer_calibrated),
    }
    metrics: dict[str, Any] = {}
    reliability_rows: list[dict[str, Any]] = []
    for model_name, (raw, calibrated) in model_probabilities.items():
        metrics[model_name] = {
            "raw": _metric_bundle(truth, raw, labels, bins),
            "calibrated": _metric_bundle(truth, calibrated, labels, bins),
        }
        prediction = labels[calibrated.argmax(axis=1)]
        per_class = pd.DataFrame(
            classification_report(
                truth, prediction, labels=labels, output_dict=True, zero_division=0
            )
        ).T.loc[labels]
        per_class.index.name = "label"
        per_class.to_csv(ROOT / "reports" / f"{model_name}_per_class_test.csv")
        _write_confusion(model_name, truth, prediction, labels)
        for row in reliability_table(truth_ids, calibrated, bins):
            reliability_rows.append({"model": model_name, **row})

    reliability = pd.DataFrame(reliability_rows)
    reliability.to_csv(ROOT / "reports" / "reliability_test.csv", index=False)
    _plot_reliability(reliability)

    baseline_prediction = labels[baseline_calibrated.argmax(axis=1)]
    transformer_prediction = labels[transformer_calibrated.argmax(axis=1)]
    protocol = evaluation["test_protocol"]
    bootstrap = paired_macro_f1_difference(
        truth,
        baseline_prediction,
        transformer_prediction,
        labels,
        int(protocol["paired_bootstrap_iterations"]),
        int(protocol["paired_bootstrap_seed"]),
        float(protocol["confidence_level"]),
    )

    champion = evaluation["champion"]
    champion_probabilities = model_probabilities[champion][1]
    champion_prediction = labels[champion_probabilities.argmax(axis=1)]
    champion_confidence, champion_margin = confidence_and_margin(champion_probabilities)
    review, reason_masks = review_mask(
        champion_probabilities,
        champion_prediction,
        float(evaluation["review_policy"]["confidence_threshold"]),
        float(evaluation["review_policy"]["top_one_top_two_margin_threshold"]),
        set(evaluation["review_policy"]["high_risk_intents"]),
    )
    automated = ~review
    selective = {
        "champion": champion,
        "coverage": float(automated.mean()),
        "selective_error": float((champion_prediction[automated] != truth[automated]).mean()),
        "automated_rows": int(automated.sum()),
        "review_rows": int(review.sum()),
        "reason_counts_nonexclusive": {
            reason: int(mask.sum()) for reason, mask in reason_masks.items()
        },
        "thresholds_selected_on_validation_only": True,
    }

    safe_text = "A newly authored customer message asks about an unfamiliar card charge."
    warmup = int(protocol["cpu_latency_warmup"])
    repetitions = int(protocol["cpu_latency_repetitions"])
    baseline_latency = _measure_baseline_latency(baseline, safe_text, warmup, repetitions)
    transformer.to("cpu")
    torch.cuda.empty_cache()
    transformer_cpu_latency = _measure_transformer_latency(
        transformer,
        tokenizer,
        safe_text,
        torch.device("cpu"),
        int(transformer_config["max_length"]),
        warmup,
        repetitions,
    )
    gpu_latency = None
    if torch.cuda.is_available():
        gpu_latency = _measure_transformer_latency(
            transformer,
            tokenizer,
            safe_text,
            torch.device("cuda"),
            int(transformer_config["max_length"]),
            warmup,
            repetitions,
        )
        transformer.to("cpu")
        torch.cuda.empty_cache()

    transformer_artifact_dir = PRIVATE / "models" / "transformer"
    result = {
        "evaluation_config_sha256": freeze_manifest["evaluation_config_sha256"],
        "official_test_rows": len(test),
        "official_test_class_support": {
            "minimum": int(test.groupby("category").size().min()),
            "maximum": int(test.groupby("category").size().max()),
        },
        "metrics": metrics,
        "paired_bootstrap": bootstrap,
        "selective_classification": selective,
        "latency": {
            "input": "one newly authored safe batch-one message",
            "baseline_cpu": baseline_latency,
            "transformer_cpu": transformer_cpu_latency,
            "transformer_gpu": gpu_latency,
            "gpu_latency_reported_only_because_measured": gpu_latency is not None,
        },
        "artifacts": {
            "baseline_bytes": (PRIVATE / "models" / "baseline" / "tfidf_logistic.joblib")
            .stat()
            .st_size,
            "transformer_weights_bytes": (transformer_artifact_dir / "model.safetensors")
            .stat()
            .st_size,
            "transformer_directory_bytes": sum(
                path.stat().st_size
                for path in transformer_artifact_dir.rglob("*")
                if path.is_file()
            ),
        },
        "runtime_memory": {
            "method": "process RSS deltas in the frozen evaluator",
            "rss_before_models_bytes": rss_before_models,
            "baseline_load_delta_bytes": max(0, rss_after_baseline - rss_before_models),
            "transformer_cpu_load_delta_bytes": max(
                0, rss_after_transformer_cpu - rss_after_baseline
            ),
        },
    }
    (ROOT / "reports" / "official_test_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    private_predictions = pd.DataFrame(
        {
            "row_id": test["row_id"],
            "true_label": truth,
            "baseline_prediction": baseline_prediction,
            "transformer_prediction": transformer_prediction,
            "champion_prediction": champion_prediction,
            "champion_confidence": champion_confidence,
            "champion_top_two_margin": champion_margin,
            "review_required_model_rules": review,
            "privacy_flag": test["privacy_flag"],
        }
    )
    private_predictions.to_csv(
        PRIVATE / "validation" / "official_test_predictions.csv", index=False
    )
    np.savez_compressed(
        PRIVATE / "validation" / "official_test_probabilities.npz",
        labels=labels,
        baseline=baseline_calibrated,
        transformer=transformer_calibrated,
    )
    state = {
        "completed": True,
        "evaluation_config_sha256": freeze_manifest["evaluation_config_sha256"],
        "results_sha256": sha256_file(ROOT / "reports" / "official_test_results.json"),
        "test_rows": len(test),
        "test_labels_reviewed_or_changed": False,
    }
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (PRIVATE / "validation" / "official_test_evaluation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

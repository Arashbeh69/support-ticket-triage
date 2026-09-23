"""Pinned transformer token audit and fine-tuning utilities."""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from support_ticket_triage.data.split import sha256_file
from support_ticket_triage.evaluation.metrics import classification_metrics


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)


def load_transformer_config(project_root: Path) -> dict[str, Any]:
    """Load the committed transformer configuration."""
    return json.loads((project_root / "configs" / "transformer.json").read_text(encoding="utf-8"))


def load_private_split(private_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only the frozen development and validation partitions."""
    split = pd.read_csv(private_root / "manifests" / "split_manifest_private.csv")
    development = split.loc[split["partition"] == "development"].reset_index(drop=True)
    validation = split.loc[split["partition"] == "validation"].reset_index(drop=True)
    return development, validation


def load_pinned_tokenizer(config: dict[str, Any], private_root: Path):
    """Load the exact model revision into a private cache."""
    cache_dir = private_root / "models" / "huggingface_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return AutoTokenizer.from_pretrained(
        config["base_model"],
        revision=config["revision"],
        cache_dir=cache_dir,
    )


def token_length_audit(project_root: Path, private_root: Path) -> dict[str, Any]:
    """Measure untruncated token lengths before selecting a maximum length."""
    config = load_transformer_config(project_root)
    development, validation = load_private_split(private_root)
    tokenizer = load_pinned_tokenizer(config, private_root)
    representation = config["representation"]
    texts = pd.concat([development[representation], validation[representation]], ignore_index=True)
    lengths = np.asarray(
        [len(tokenizer.encode(text, add_special_tokens=True, truncation=False)) for text in texts],
        dtype=int,
    )
    percentiles = {
        str(percentile): float(np.percentile(lengths, percentile))
        for percentile in (50, 90, 95, 99, 99.5, 99.9, 100)
    }
    result = {
        "base_model": config["base_model"],
        "revision": config["revision"],
        "licence": config["licence"],
        "representation": representation,
        "rows": int(len(lengths)),
        "minimum": int(lengths.min()),
        "mean": float(lengths.mean()),
        "percentiles": percentiles,
        "counts_above": {
            str(limit): int((lengths > limit).sum()) for limit in (32, 48, 64, 96, 128)
        },
        "official_test_used": False,
    }
    private_validation = private_root / "validation"
    private_validation.mkdir(parents=True, exist_ok=True)
    for destination in (
        private_validation / "token_length_audit.json",
        project_root / "reports" / "token_length_audit.json",
    ):
        destination.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result


def _encode(tokenizer, texts: pd.Series, max_length: int) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        texts.tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}


def _make_loader(
    encoded: dict[str, torch.Tensor],
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def predict_logits(
    model,
    tokenizer,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return ordered logits for texts without retaining text in the output."""
    encoded = _encode(tokenizer, pd.Series(texts), max_length)
    dummy_labels = torch.zeros(len(texts), dtype=torch.long)
    loader = _make_loader(encoded, dummy_labels, batch_size, False, seed=0)
    logits, _ = _predict(model, loader, device)
    return logits


@torch.no_grad()
def _predict(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for input_ids, attention_mask, targets in loader:
        output = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
        logits.append(output.logits.float().cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def train_transformer(project_root: Path, private_root: Path) -> dict[str, Any]:
    """Fine-tune the pinned model and select epochs only by validation macro F1."""
    config = load_transformer_config(project_root)
    if "max_length" not in config:
        raise ValueError("Run the token audit and commit a measured max_length before training")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; record the measured constraint before changing plan"
        )
    seed = int(config["random_seed"])
    set_reproducible_seed(seed)
    development, validation = load_private_split(private_root)
    labels = json.loads((project_root / "configs" / "labels.json").read_text(encoding="utf-8"))
    label_to_id = {label: index for index, label in enumerate(labels)}
    tokenizer = load_pinned_tokenizer(config, private_root)
    representation = config["representation"]
    max_length = int(config["max_length"])
    train_encoded = _encode(tokenizer, development[representation], max_length)
    validation_encoded = _encode(tokenizer, validation[representation], max_length)
    train_labels = torch.tensor(
        development["category"].map(label_to_id).to_numpy(), dtype=torch.long
    )
    validation_labels = torch.tensor(
        validation["category"].map(label_to_id).to_numpy(), dtype=torch.long
    )
    batch_size = int(config["per_device_batch_size"])
    train_loader = _make_loader(train_encoded, train_labels, batch_size, True, seed)
    validation_loader = _make_loader(
        validation_encoded, validation_labels, batch_size * 2, False, seed
    )
    cache_dir = private_root / "models" / "huggingface_cache"
    model = AutoModelForSequenceClassification.from_pretrained(
        config["base_model"],
        revision=config["revision"],
        cache_dir=cache_dir,
        num_labels=len(labels),
        id2label={index: label for index, label in enumerate(labels)},
        label2id=label_to_id,
    )
    device = torch.device("cuda")
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    accumulation = int(config["gradient_accumulation_steps"])
    total_steps = math.ceil(len(train_loader) / accumulation) * int(config["epochs"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(config["warmup_ratio"])),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config["precision"] == "fp16")
    output_dir = private_root / "models" / "transformer"
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_macro_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()
    peak_allocated = 0

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        for step, (input_ids, attention_mask, targets) in enumerate(train_loader, start=1):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                output = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    labels=targets.to(device),
                )
                loss = output.loss / accumulation
            scaler.scale(loss).backward()
            epoch_loss += float(loss.item()) * accumulation
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated()))

        validation_logits, validation_ids = _predict(model, validation_loader, device)
        validation_predictions = validation_logits.argmax(axis=1)
        macro_f1 = float(f1_score(validation_ids, validation_predictions, average="macro"))
        epoch_record = {
            "epoch": epoch,
            "train_loss": epoch_loss / len(train_loader),
            "validation_macro_f1": macro_f1,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(epoch_record)
        (private_root / "logs").mkdir(parents=True, exist_ok=True)
        (private_root / "logs" / "transformer_training.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        if macro_f1 > best_macro_f1 + 1e-6:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            stale_epochs = 0
            model.save_pretrained(output_dir, safe_serialization=True)
            tokenizer.save_pretrained(output_dir)
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["early_stopping_patience"]):
                break

    best_model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    validation_logits, validation_ids = _predict(best_model, validation_loader, device)
    probabilities = torch.softmax(torch.from_numpy(validation_logits), dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    metrics = classification_metrics(
        np.asarray([labels[index] for index in validation_ids]),
        np.asarray([labels[index] for index in predictions]),
        probabilities,
        np.asarray(labels),
    )
    artifact_path = output_dir / "model.safetensors"
    result = {
        "model": config["base_model"],
        "revision": config["revision"],
        "licence": config["licence"],
        "device": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "max_length": max_length,
        "learning_rate": config["learning_rate"],
        "epochs_requested": config["epochs"],
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "per_device_batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": batch_size * accumulation,
        "warmup_ratio": config["warmup_ratio"],
        "weight_decay": config["weight_decay"],
        "max_grad_norm": config["max_grad_norm"],
        "precision": config["precision"],
        "peak_cuda_bytes": peak_allocated,
        "training_seconds": time.perf_counter() - started,
        "official_test_used": False,
        "validation": metrics,
        "history": history,
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_bytes": artifact_path.stat().st_size,
        "config_sha256": sha256_file(project_root / "configs" / "transformer.json"),
    }
    for destination in (
        private_root / "validation" / "transformer_validation.json",
        project_root / "reports" / "transformer_validation.json",
    ):
        destination.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    public_manifest = {
        "relative_private_path": "models/transformer/model.safetensors",
        "sha256": result["artifact_sha256"],
        "bytes": result["artifact_bytes"],
        "stored_in_git": False,
        "base_model": config["base_model"],
        "revision": config["revision"],
        "licence": config["licence"],
    }
    (project_root / "manifests" / "transformer_artifact.json").write_text(
        json.dumps(public_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result

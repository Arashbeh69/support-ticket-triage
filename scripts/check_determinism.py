"""Repeat one identical GPU training step to verify the configured boundary."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd
import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from support_ticket_triage.models.transformer import set_reproducible_seed

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT.with_name(f"{ROOT.name}-private")


def _trial(config: dict, encoded: dict[str, torch.Tensor], targets: torch.Tensor) -> dict:
    seed = int(config["random_seed"])
    set_reproducible_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForSequenceClassification.from_pretrained(
        config["base_model"],
        revision=config["revision"],
        cache_dir=PRIVATE / "models" / "huggingface_cache",
        local_files_only=True,
        num_labels=77,
    ).cuda()
    optimizer = AdamW(model.parameters(), lr=float(config["learning_rate"]))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(
            input_ids=encoded["input_ids"].cuda(),
            attention_mask=encoded["attention_mask"].cuda(),
            labels=targets.cuda(),
        )
    output.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
    optimizer.step()
    torch.cuda.synchronize()
    digest = hashlib.sha256(
        model.classifier.out_proj.weight.detach().cpu().numpy().tobytes()
    ).hexdigest()
    result = {
        "loss": float(output.loss.item()),
        "classifier_weight_sha256_after_step": digest,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
    }
    del model, optimizer, output
    torch.cuda.empty_cache()
    return result


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this boundary-specific repeatability check")
    config = json.loads((ROOT / "configs" / "transformer.json").read_text(encoding="utf-8"))
    split = pd.read_csv(PRIVATE / "manifests" / "split_manifest_private.csv")
    texts = (
        split.loc[split["partition"] == "development", config["representation"]]
        .head(int(config["per_device_batch_size"]))
        .tolist()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"],
        revision=config["revision"],
        cache_dir=PRIVATE / "models" / "huggingface_cache",
        local_files_only=True,
    )
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=int(config["max_length"]),
        return_tensors="pt",
    )
    targets = torch.arange(len(texts), dtype=torch.long) % 77
    first = _trial(config, encoded, targets)
    second = _trial(config, encoded, targets)
    result = {
        "scope": "two identical one-step GPU trials on the pinned stack and device",
        "strict_repeatability_observed": (
            first["loss"] == second["loss"]
            and first["classifier_weight_sha256_after_step"]
            == second["classifier_weight_sha256_after_step"]
        ),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "flash_attention_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "memory_efficient_attention_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_attention_enabled": torch.backends.cuda.math_sdp_enabled(),
        "tf32_matmul_enabled": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn_enabled": torch.backends.cudnn.allow_tf32,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "trials": [first, second],
        "limitation": (
            "This does not promise bitwise identity across other hardware or software stacks."
        ),
    }
    for destination in (
        PRIVATE / "validation" / "determinism_repeatability.json",
        ROOT / "reports" / "determinism_repeatability.json",
    ):
        destination.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

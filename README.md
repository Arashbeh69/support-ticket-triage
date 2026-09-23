# Banking support-intent triage

An evidence-first NLP project for routing short English banking-support messages
across BANKING77's 77 intents. It compares a word-and-character TF-IDF logistic
regression baseline with a pinned, fine-tuned DistilRoBERTa challenger, then
freezes calibration and human-review rules before a one-time official test.

The calibrated baseline is the selected operational model. On the 3,080-row
official test it reached **0.9077 macro F1**, **0.9075 accuracy**, and **0.9721
top-3 accuracy**. DistilRoBERTa reached 0.9083 macro F1, but its +0.0007 point
difference had a paired-bootstrap 95% interval of −0.0088 to +0.0105 and did not
clear the predeclared validation gain rule. The baseline artifact is 10.6 MB
versus 328.7 MB for transformer weights and had measured CPU batch-one p50/p95
latency of 8.89/10.91 ms.

This is a portfolio system, not a production banking service. It is not an
automated resolution engine, an open-set guarantee, financial advice, fraud
detection, or permission to act on an account.

## What is included

- Exact source and model revision contracts, SHA-256 manifests, and licensing.
- Aggregate privacy, duplicate, leakage, split, and class-distribution evidence.
- Leakage-safe development/validation splitting with the official test held out.
- Reproducible sparse baseline and deterministic-GPU transformer training code.
- Frozen temperature calibration, review thresholds, and one-time test protocol.
- Per-class metrics, confusion matrices, reliability plots, selective-risk
  analysis, paired bootstrap uncertainty, and aggregate error analysis.
- Local CLI, FastAPI service, static browser demo, CPU baseline Docker image,
  and CI checks that do not retrain the transformer.

Raw utterances, row-level audit records, predictions, logs, and model weights
are intentionally excluded from Git. Public reports contain aggregate or
text-free evidence only.

## Key evidence

| Result | Baseline | DistilRoBERTa |
| --- | ---: | ---: |
| Validation macro F1 | 0.9019 | 0.9057 |
| Official-test macro F1 | 0.9077 | 0.9083 |
| Official-test weighted F1 | 0.9077 | 0.9083 |
| Official-test top-3 accuracy | 0.9721 | 0.9760 |
| Calibrated test log loss | 0.3348 | 0.3517 |
| Calibrated test ECE, 15 equal-width bins | 0.0086 | 0.0164 |
| Artifact size | 10.6 MB | 328.7 MB weights |
| CPU batch-one p50 / p95 | 8.89 / 10.91 ms | 59.58 / 93.16 ms |

The validation-selected review policy automatically retains 81.27% of official
test rows at 4.59% selective error. Configured high-risk intents, detected
sensitive patterns, malformed inputs, and obvious domain mismatches always go
to review. This is a measured portfolio policy, not production validation.

See the [model card](docs/MODEL_CARD.md), [data card](docs/DATA_CARD.md),
[error analysis](reports/ERROR_ANALYSIS.md), and
[reproducibility guide](docs/REPRODUCIBILITY.md) for definitions and caveats.

## Run locally

Python 3.12 is the verified environment. The lightweight path serves the frozen
baseline and does not require transformer dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[api,dev]"
$env:SUPPORT_TRIAGE_PRIVATE_ROOT = "D:\path\to\support-ticket-triage-ml-private"
.\.venv\Scripts\support-triage.exe metadata
.\.venv\Scripts\support-triage.exe predict --text "I need help with a card payment"
.\.venv\Scripts\uvicorn.exe support_ticket_triage.api:app --host 127.0.0.1 --port 8000 --no-access-log
```

Endpoints are `GET /health`, `GET /metadata`, and `POST /predict`. The local
demo is at `http://127.0.0.1:8000/demo/`. Prediction responses include the top
three intents, calibrated confidence, routing group, and nonexclusive review
reasons. Application logs exclude raw message text.

For the container, mount the private artifact sibling read-only:

```powershell
.\scripts\docker_smoke.ps1 -PrivateRoot "D:\path\to\support-ticket-triage-ml-private"
```

The image runs as a non-root user, includes a health check, and contains no
dataset rows or model artifact. The current frozen baseline image was built and
smoke-tested locally.

## Reproduce the evidence

The authoritative source is PolyAI's repository pinned at
`57ec275d8078af65b7731c2a98be812d844a6d6b`; DistilRoBERTa is pinned at
`fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b`. Exact hashes and acquisition URLs
are in `manifests/`. Start with [the full reproduction instructions](docs/REPRODUCIBILITY.md).

The official-test script is deliberately guarded by a private access record.
Do not rerun or retune against the test. Transformer weights remain outside
ordinary Git history and must be reproduced or supplied separately in the
private artifact directory.

## Licence and attribution

Project code is MIT licensed. BANKING77 is CC BY 4.0 and has separate
attribution requirements; see [DATA_LICENSE.md](DATA_LICENSE.md). The upstream
licence permits reuse, but it is not evidence of privacy review, which is why
source text is not redistributed here.

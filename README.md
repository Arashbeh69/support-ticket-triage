# Banking support-intent triage

An evidence-first NLP project comparing a word-and-character TF-IDF logistic
regression baseline with a fine-tuned compact transformer on BANKING77.

The task is English, banking-specific, single-query intent classification. It
is not complete help-desk ticket processing, automated resolution, financial
advice, fraud detection, or a production banking system.

Implementation is in progress. Raw customer queries and model checkpoints are
kept in a private sibling directory and are not part of this repository.

## Source contract

The original PolyAI repository is pinned at
`57ec275d8078af65b7731c2a98be812d844a6d6b`. See
[`DATA_LICENSE.md`](DATA_LICENSE.md) and
[`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md).

## Reproduction boundary

The planned public workflow will reproduce source verification, privacy and
duplicate audits, the leakage-aware split, the classical baseline, evaluation,
and inference. Transformer weights remain outside ordinary Git history.


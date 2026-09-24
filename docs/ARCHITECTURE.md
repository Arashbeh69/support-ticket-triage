# Architecture

## Boundary and flow

```text
Pinned BANKING77 CSVs (private)
  -> source verification
  -> private normalized/redacted representations
  -> aggregate privacy and duplicate audit
  -> grouped development/validation split + sealed official test
  -> baseline and transformer artifacts (private)
  -> frozen calibration and review configuration
  -> text-free aggregate evaluation evidence (public)
  -> local CLI / FastAPI / browser demo
```

The public repository contains implementation, configuration, tests, hashes,
aggregate reports, figures, and deployment scaffolding. The private sibling
contains every source utterance, row-level audit record, model artifact,
prediction record, training log, and checkpoint.

## Package responsibilities

- `data.source` acquires and verifies the pinned upstream files.
- `text` provides deterministic normalization, sensitive-pattern detection, and
  redaction.
- `data.audit` measures privacy, exact/template duplication, near-duplicate
  sensitivity, and training-side test overlap.
- `data.split` creates a stratified grouped development/validation split without
  moving or selecting the official test.
- `models.baseline` fits word/character TF-IDF and multinomial logistic
  regression.
- `models.transformer` audits token lengths and fine-tunes the pinned compact
  encoder with controlled seeds and CUDA kernels.
- `evaluation` owns metrics, temperature scaling, review-threshold selection,
  reliability, and paired bootstrap uncertainty.
- `inference` validates input, scores the frozen champion, returns top-three
  candidates, and independently attaches nonexclusive confidence, margin,
  privacy, domain-keyword, and high-risk review reasons. The transparent
  domain-keyword rule is not trained OOD detection, and high confidence cannot
  override another applicable rule.
- `api` exposes health, metadata, and prediction endpoints without raw-text
  logging.

## Artifact dependency rule

Model weights are never committed or copied into ordinary container layers.
Each artifact has a public manifest with relative private path, byte size, and
SHA-256. Local serving resolves the private sibling or an explicit
`SUPPORT_TRIAGE_PRIVATE_ROOT`; Docker expects the artifact directory as a
read-only mount.

The evaluation freeze hashes preprocessing files, split manifests, both selected
model artifacts, calibration parameters, review policy, and the complete test
protocol before official test access.

`configs/evaluation.json` is an immutable historical evaluation record whose
hash defines the reported model version. It therefore retains the original
pre-correction domain-reason spelling. The active runtime contract is defined
by `configs/review_policy.json` and `inference`: runtime predictions emit
`no_domain_keyword_detected`, never the historical spelling.

Runtime predictions and metadata expose `review_policy_version`, a 16-character
prefix of the canonical review-policy JSON's SHA-256. Canonicalization makes it
stable across line endings and JSON key order. It changes independently of the
immutable evaluation-derived `model_version`, making model and operational
policy revisions separately traceable.

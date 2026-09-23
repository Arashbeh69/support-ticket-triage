# Monitoring design

This is a proposed production monitoring design, not evidence from a live
deployment. Raw user text should not be logged by default.

## Online service health

- Request count, sanitized 4xx/5xx rate, timeout rate, and model-load failures.
- Batch-one p50/p95/p99 latency by model and version.
- Process memory, CPU saturation, restart count, and artifact checksum status.
- Trace every aggregate to application version, evaluation-config hash, and
  model-artifact hash.

## Input and privacy signals

- Character, byte, and token-length distributions and truncation rate.
- Control-character and malformed-input rejection rate.
- Sensitive-pattern detection rate by rule, reported only in aggregate.
- Domain-mismatch and vocabulary/OOV proxies. Embedding drift is justified only
  if the transformer is deployed and a stable, privacy-reviewed reference
  representation is retained.

## Prediction and review behavior

- Predicted-intent and routing-group distribution against a dated reference.
- Calibrated-confidence and top-two-margin distributions.
- Automated coverage, abstention rate, and nonexclusive review-reason counts.
- High-risk-intent prediction and review rates.
- Calibration drift on later human-labeled samples using the same declared ECE
  definition, plus log loss and Brier score.

## Quality evidence

- Macro and weighted F1, accuracy, top-3 accuracy, and per-class precision,
  recall, F1, and support on later independently labeled samples.
- High-risk false-routing count and rate, reviewed separately from aggregate
  accuracy.
- Confusion-pair movement, support changes, and quality by input length,
  confidence bin, privacy detection, and template/duplicate status where lawful.

## Response principles

Alert thresholds require operational validation. A drift alarm should trigger
data and labeling review, not automatic retraining. Any retraining needs a new
source contract, privacy audit, leakage-safe split, calibration freeze, artifact
hash, and independent evaluation. Retain only minimum aggregate telemetry and
approved human labels; never add raw tickets to training silently.

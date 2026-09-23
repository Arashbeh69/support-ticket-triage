# Model card

## Model decision

The deployed portfolio candidate is a calibrated word-and-character TF-IDF
multinomial logistic-regression model. A pinned DistilRoBERTa model is retained
as a challenger. The baseline was selected before official-test access because
the transformer improved validation macro F1 by only 0.00376, below the frozen
0.010 minimum gain; the weighted-F1 guardrail was also checked.

## Task and intended use

Input is one short English banking-support message. Output is one of the exact
77 BANKING77 intent labels, a project-defined routing group, calibrated
confidence, the top three candidates, and review reasons. Intended use is a
local portfolio demonstration of assisted triage. It is not approved for live
customer routing, autonomous action, financial advice, security decisions,
fraud adjudication, or identity verification.

## Data and evaluation protocol

The source contains 10,003 official training and 3,080 official test rows. A
privacy and leakage audit quarantined 14 credential-value candidates and
excluded 183 training-side near overlaps at a 0.98 threshold. Grouped
stratification created 8,416 development and 1,404 validation rows with all 77
labels and no duplicate group crossing. Model choice, scalar temperature
calibration, and review thresholds used validation only. The official test was
accessed once after the evaluation configuration was hashed and frozen.

## Performance

| Metric | Calibrated baseline | Calibrated transformer |
| --- | ---: | ---: |
| Macro F1 | 0.9077 | 0.9083 |
| Weighted F1 | 0.9077 | 0.9083 |
| Accuracy | 0.9075 | 0.9084 |
| Top-3 accuracy | 0.9721 | 0.9760 |
| Log loss | 0.3348 | 0.3517 |
| Multiclass Brier | 0.1368 | 0.1454 |
| ECE, 15 equal-width bins | 0.0086 | 0.0164 |

The paired 3,000-resample bootstrap interval for transformer-minus-baseline test
macro F1 is −0.00876 to +0.01048 around a +0.00066 point estimate. It does not
establish a transformer advantage.

The baseline review policy uses a 0.4 top-one/top-two probability margin and
always reviews configured high-risk intents. On the official test it covered
81.27% of rows with 4.59% error among automated rows. Confidence 0.0 is not a
mistake: the validation search found margin plus mandatory review rules was the
highest-coverage combination satisfying the 5% selective-error constraint.

## Operational measurements

The 10,556,044-byte baseline artifact measured 8.89 ms p50 and 10.91 ms p95 for
CPU batch-one inference. Transformer weights are 328,722,980 bytes; measured
latency was 59.58/93.16 ms p50/p95 on CPU and 11.46/16.47 ms on the available
GPU. These are local measurements on one newly authored safe input, not service
capacity or production SLO evidence.

## Error analysis

The baseline made 285 intent errors and 91 broader routing-group errors. Major
confusions include pending transfer versus balance not updated, card acceptance
versus card not working, and several identity-verification distinctions.
Heuristic review queues found 56 ambiguity proxies among errors, but those
flags are not human adjudication and no official label was changed. Public
reports contain no source utterance text.

## Safety and limitations

- There is no explicit unknown-intent class or validated open-set detector.
- Confidence is not correctness, and calibration may drift in live traffic.
- The balanced benchmark does not represent production prevalence or impact.
- Privacy detection is rule based and cannot prove anonymization.
- English, short-message results do not transfer automatically to other
  languages, channels, conversation histories, or banking populations.
- Project-created routing groups have no source or business-process ground
  truth.
- High-risk review rules reduce automation but do not prove all harmful errors
  are caught.

Any real deployment would require privacy, security, fairness, operational,
human-factors, open-set, drift, and domain-specific validation.

## Reproducibility

The source revision, model revision, split, preprocessing, artifacts,
calibration, review policy, and evaluation configuration have public hashes.
Raw text, predictions, training logs, and model artifacts remain private. The
first transformer attempt was rejected before epoch one because deterministic
CUDA requirements were not satisfied; the corrected run and a two-trial
one-step repeatability check are documented. Reproducibility is scoped to the
pinned stack and hardware boundary, not arbitrary platforms.

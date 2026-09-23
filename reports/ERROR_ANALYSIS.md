# Structured error analysis

The frozen champion makes 285 intent errors across 3,080 official test rows and 91 errors that cross a project-defined routing-group boundary.

## Dimensions

Aggregate CSVs break error rate down by operational group, development support, text length, calibrated-confidence bin, duplicate/template status, a word-OOV spelling proxy, redaction status, and high-risk consequence. Raw text and row-level predictions remain private.

## Semantic review queues

Among errors, 1 have at most three words, 56 have a top-two margin below 0.10, and 1 are high-confidence duplicate/template cases queued for possible annotation review. These are review heuristics, not adjudications. No official test label was changed.

## Operational consequence

50 errors involve a configured high-risk true or predicted intent. Those predictions are always sent to human review by the portfolio policy. Confidence and abstention do not establish safe open-set behavior.

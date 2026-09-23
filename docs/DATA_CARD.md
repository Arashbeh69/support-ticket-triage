# Data card

## Source

BANKING77 is taken from PolyAI's original `task-specific-datasets` repository at
commit `57ec275d8078af65b7731c2a98be812d844a6d6b`. The verified files contain
10,003 training rows, 3,080 official test rows, and 77 English banking-support
intent labels. The source is licensed CC BY 4.0. Exact URLs and SHA-256 hashes
are in `manifests/source_manifest.json`.

## Intended project use

This repository evaluates short, single-query, closed-set intent
classification. It does not provide priority, agent, resolution, or business
routing ground truth. Six broader routing groups are project-created metadata,
not source annotations.

## Privacy treatment

The upstream documentation does not describe a privacy review. Raw and derived
utterances therefore stay private even though the licence permits reuse. A
rule-based scan found 14 credential-value candidates, which were redacted for
modeling and quarantined for private review. No raw flagged example is published.
The scanner is a risk-reduction layer, not proof of anonymization.

## Duplication and leakage

The audit found 26 exact normalized duplicate groups containing 52 training
rows, including one conflicting-label group, and five generalized template
groups. A provisional 0.92 near-duplicate threshold was rejected because it
would remove 779 training rows as test overlaps. The frozen 0.98 threshold
identified 207 training candidate pairs, 195 cross-split pairs, and 183 training
rows excluded for official-test overlap.

## Model-selection partitions

Grouped stratification produces 8,416 development and 1,404 validation rows;
all 77 labels remain in both, and no duplicate group crosses the boundary. The
official 3,080-row test remains fixed at 40 examples per label and is not used
for preprocessing, model, calibration, or threshold selection.

## Known limitations

The source documentation is incomplete about collection, annotation, annotator
agreement, demographics, privacy review, and temporal coverage. There is no
explicit unknown-intent class or conversation context. English banking queries
do not establish performance for other languages, domains, channels, or live
customer populations. Similar wording is intrinsic to this task, so duplicate
rules can both miss paraphrases and over-group legitimate examples.

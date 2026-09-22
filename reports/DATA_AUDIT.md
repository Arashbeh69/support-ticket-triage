# BANKING77 data audit

The audit was run against PolyAI commit
`57ec275d8078af65b7731c2a98be812d844a6d6b`. Only aggregate findings are
published here; no source utterance is reproduced.

## Source and class checks

- 10,003 training rows and 3,080 official test rows were read from the pinned
  CSV files.
- Both splits contain all 77 exact source labels.
- There are no empty text fields.
- Training support ranges from 35 to 187 rows per class.
- The official test set has 40 rows per class.
- The six project-created routing groups cover every source label exactly once.

## Privacy screening

Fourteen rows triggered the conservative credential-value rule and were placed
in the private quarantine for review. No raw flagged example is public. The
scan did not treat every occurrence of words such as `PIN` or every number as
personal information; a credential keyword had to be followed by a value-like
string.

This automated result is not proof that the remaining text is free of personal
information. The rule, reason, span, severity, hashed match, and disposition are
retained privately for every detection.

## Duplication and leakage

- 26 normalized exact-duplicate groups contain 52 training rows.
- One exact-duplicate group contains conflicting labels.
- Five generalized template groups contain more than one distinct normalized
  wording.
- Character 3–5-gram TF-IDF found 207 nearest-neighbour candidate pairs at the
  frozen 0.98 cosine threshold.
- Cross-split audit found 195 candidate pair links, involving 183 training rows.
  Those training-side rows are excluded before development/validation splitting;
  official test rows are never moved.

The initial 0.92 threshold was rejected because it would have removed 779
training rows. For these short queries, ordinary semantic similarity is not
sufficient evidence of duplication. The threshold change was made using
training-data sensitivity counts before model selection.

Machine-readable evidence is in `data_audit_summary.json`,
`class_distribution.csv`, and `../manifests/source_manifest.json`.


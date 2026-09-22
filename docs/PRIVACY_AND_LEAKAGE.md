# Privacy and leakage controls

The BANKING77 licence permits reuse, but the source documentation does not
describe a privacy review. This project therefore treats every source utterance
as private until automated findings have been reviewed.

## Representations

- **Raw:** immutable pinned CSV files in the private sibling.
- **Normalized audit:** Unicode NFKC plus whitespace normalization, with raw
  text retained privately for traceability.
- **Modelling:** normalized wording with punctuation and case preserved.
- **Redacted candidate:** rule-based placeholders for measured comparison, not
  an automatically assumed replacement for the benchmark.
- **Quarantine:** high-confidence credential, IBAN, email, or Luhn-valid
  card-number candidates, retained only in the private sibling.

Detection is intentionally evidence-based. Generic words such as `PIN`,
`password`, and `card` are not redacted unless a rule finds an accompanying
value. Every detection records the rule, reason, severity, disposition, span,
and a hash of the matched value.

## Duplicate and leakage policy

Exact duplicates use a case-folded, punctuation-insensitive key. Templates use
the same key with numeric and currency slots generalized. Near duplicates use
character 3–5-gram TF-IDF cosine similarity. The selected threshold is fixed
from training-data sensitivity counts before cross-split auditing.

Training examples that overlap the official test set at or above the frozen
threshold are excluded from development. Test items are never moved or used to
select preprocessing, features, thresholds, or models. The official labels are
read only for source-contract verification and the final frozen evaluation.


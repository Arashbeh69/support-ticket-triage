# Publication verification

This record is scoped to public-repository verification on 25 September 2026.
`deployment_verification.json` remains unchanged historical evidence from the
final local build before any remote repository existed; its statement that
remote CI had not run was accurate at that time.

The initial public GitHub Actions run
[`36066513879`](https://github.com/Arashbeh69/support-ticket-triage/actions/runs/36066513879)
exposed a cross-platform byte-normalization defect. Git stored LF-normalized
blobs for two immutable records whose frozen SHA-256 values describe their
historical CRLF bytes. Model training, model selection, calibration, metrics,
artifact hashes, and protected official-test results were not implicated.

The corrective publication change stores only `manifests/split_manifest.csv`
and `configs/evaluation.json` verbatim as CRLF historical records, enforces LF
for ordinary text, and verifies both checkout and Git-index bytes. Subsequent
workflow status is authoritative in the repository's GitHub Actions history.

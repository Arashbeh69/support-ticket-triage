# BANKING77 source contract

## Authority

The authoritative source is PolyAI's `task-specific-datasets` repository at
commit `57ec275d8078af65b7731c2a98be812d844a6d6b`. Acquisition uses raw GitHub
URLs containing this full commit rather than a moving branch name.

Expected files:

| File | Expected records | Expected fields |
| --- | ---: | --- |
| `train.csv` | 10,003 | `text`, `category` |
| `test.csv` | 3,080 | `text`, `category` |
| `categories.json` | 77 labels | JSON array |
| `LICENSE` | n/a | CC BY 4.0 text |

Counts are source expectations, not accepted evidence. The acquisition audit
must independently verify schema, record counts, source-label membership,
hashes, and licence content before modelling.

## Label policy

All 77 source strings remain byte-for-byte traceable, including
`Refund_not_showing_up` and `reverted_card_payment?`. Human-friendly names are
presentation aliases only. The six routing groups in
`configs/routing_groups.json` are project-created metadata and are not supplied
by PolyAI.

## Public/private boundary

Raw and normalized utterances, privacy findings at row level, split manifests
containing text, training logs, and model weights belong in
`support-ticket-triage-ml-private`. The public repository may contain aggregate
statistics, text-free row IDs, hashes, source code, tests, and documentation.


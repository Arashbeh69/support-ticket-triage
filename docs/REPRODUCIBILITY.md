# Reproducibility

## Environment

The project targets Python 3.12. Core, API, transformer, and development
dependencies are pinned in `pyproject.toml`. The verified local training
environment uses PyTorch 2.8.0 with CUDA 12.8 on a 4 GB GTX 1650 Ti. CPU-only
inference remains supported.

## Fresh-clone baseline and local demo path

The baseline artifact is deliberately absent from Git. From a fresh clone,
create the environment, acquire the pinned source into the private sibling,
run the privacy/leakage audit and grouped split, and rebuild the sparse baseline:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[api,dev]"
.\.venv\Scripts\python.exe scripts\acquire_source.py
.\.venv\Scripts\python.exe scripts\run_audit.py
.\.venv\Scripts\python.exe scripts\build_split.py
.\.venv\Scripts\python.exe scripts\train_baseline.py
```

Acquisition requires network access. Later steps use local pinned artifacts. Do
not run `scripts\evaluate_official_test.py`: the published aggregate evidence
already records the one-time protected evaluation. BANKING77 is CC BY 4.0; keep
the included attribution and upstream licence with any permitted source use.

The scripts create `support-ticket-triage-ml-private` beside the clone and keep
source rows, audit records, split text, and the fitted artifact out of the public
repository. The rebuilt artifact belongs at
`models\baseline\tfidf_logistic.joblib` under that private sibling. Compare its
byte size and SHA-256 with `manifests/baseline_artifact.json` before serving.

With the artifact available, run the CLI and local service:

```powershell
.\.venv\Scripts\support-triage.exe metadata
.\.venv\Scripts\support-triage.exe predict --text "I forgot my card PIN"
.\.venv\Scripts\uvicorn.exe support_ticket_triage.api:app --host 127.0.0.1 --port 8000 --no-access-log
```

Open `http://127.0.0.1:8000/demo/` in a browser. The demo calls only the local
API. Metadata and predictions report both the immutable model version and the
active review-policy version.

Run the public checks without training or model artifacts:

```powershell
$testTmp = Join-Path $env:TEMP ("support-ticket-triage-tests-" + [guid]::NewGuid())
.\.venv\Scripts\python.exe -m pytest --basetemp $testTmp
.\.venv\Scripts\python.exe -m ruff check .
```

## Baseline artifact distribution

Keep the 10.6 MB baseline artifact excluded from the repository and rebuild it
locally for now. Although its size is modest and BANKING77 permits reuse under
CC BY 4.0 with attribution, a fitted TF-IDF/joblib artifact should not be
distributed until a dedicated memorization/privacy review and serialized-model
security review are complete. A later GitHub Release is preferable to ordinary
Git history if approved: publish the exact hash, byte size, licence attribution,
Python/scikit-learn compatibility, model card, and a warning never to load an
untrusted joblib file. Transformer weights remain excluded.

## Transformer path

Install the pinned transformer dependencies, verify CUDA or accept CPU cost,
then run token audit before training:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[transformer,dev]"
.\.venv\Scripts\python.exe scripts\audit_token_lengths.py
.\.venv\Scripts\python.exe scripts\train_transformer.py
```

The configuration sets Python, NumPy, PyTorch, and CUDA seeds; the CuBLAS
workspace is configured before PyTorch import; nondeterministic attention kernels
are disabled; DataLoader workers remain zero; and deterministic algorithms are
required. Reproducibility is scoped to the pinned software, data, hardware, and
CUDA configuration rather than promised across arbitrary platforms.

## Artifact verification and serving

Public manifests give expected private artifact paths, byte sizes, and hashes.
Verify them before CLI, API, or Docker use. Local serving loads one frozen model
at startup:

```powershell
support-triage metadata
support-triage predict --text "A newly authored local test message about a card charge"
uvicorn support_ticket_triage.api:app --host 127.0.0.1 --port 8000 --no-access-log
```

The demo is available at `http://127.0.0.1:8000/demo/` and sends text only to the
local API. The Docker image contains code and configuration but no data or model
artifact; mount the private sibling read-only at `/artifacts`.

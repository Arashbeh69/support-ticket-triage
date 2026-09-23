# Reproducibility

## Environment

The project targets Python 3.12. Core, API, transformer, and development
dependencies are pinned in `pyproject.toml`. The verified local training
environment uses PyTorch 2.8.0 with CUDA 12.8 on a 4 GB GTX 1650 Ti. CPU-only
inference remains supported.

## Lightweight public path

Create an environment, acquire the pinned source into the private sibling, run
the audit and split, then reproduce the sparse baseline:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[api,dev]"
.\.venv\Scripts\python.exe scripts\acquire_source.py
.\.venv\Scripts\python.exe scripts\run_audit.py
.\.venv\Scripts\python.exe scripts\build_split.py
.\.venv\Scripts\python.exe scripts\train_baseline.py
.\.venv\Scripts\python.exe -m pytest
```

Acquisition requires network access. Later steps use local pinned artifacts. Do
not rerun an official test evaluation after its private access record exists.

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

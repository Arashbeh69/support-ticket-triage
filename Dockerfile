FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SUPPORT_TRIAGE_PRIVATE_ROOT=/artifacts

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --home-dir /nonexistent app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY manifests ./manifests
COPY demo ./demo

RUN python -m pip install --no-cache-dir -e ".[api]"

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "support_ticket_triage.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]

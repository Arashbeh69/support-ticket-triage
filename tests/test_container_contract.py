"""Static checks for the CPU-safe, non-root container boundary."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_is_pinned_non_root_and_has_healthcheck() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.startswith("FROM python:3.12.11-slim-bookworm")
    assert "USER app" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert '".[api]"' in dockerfile
    assert "transformer" not in dockerfile
    assert "--no-access-log" in dockerfile


def test_docker_context_excludes_private_and_model_material() -> None:
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    required = {"data", "models", "runs", "logs", "*.safetensors", "*.joblib", ".env"}
    assert required.issubset(set(ignored))

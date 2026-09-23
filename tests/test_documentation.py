"""Local Markdown-link and public-path checks."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")


def test_local_markdown_links_resolve() -> None:
    for document in ROOT.rglob("*.md"):
        if ".venv" in document.parts or ".pytest-tmp" in document.parts:
            continue
        for target in MARKDOWN_LINK.findall(document.read_text(encoding="utf-8")):
            destination = target.strip().split("#", 1)[0]
            if not destination or "://" in destination or destination.startswith("mailto:"):
                continue
            assert (document.parent / destination).resolve().exists(), (
                f"broken local link in {document.relative_to(ROOT)}: {target}"
            )


def test_public_docs_do_not_expose_local_user_paths_or_student_email() -> None:
    forbidden = ("C:\\Users\\Arash", "D:\\PortfolioFast", "tuwien.ac.at")
    for document in ROOT.rglob("*.md"):
        if ".venv" in document.parts or ".pytest-tmp" in document.parts:
            continue
        content = document.read_text(encoding="utf-8")
        assert not any(value.casefold() in content.casefold() for value in forbidden)

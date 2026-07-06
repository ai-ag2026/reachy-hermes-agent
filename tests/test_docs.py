from pathlib import Path


def test_public_release_docs_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "README.md",
        root / "LICENSE-APACHE",
        root / "LICENSE-MIT",
        root / "CONTRIBUTING.md",
        root / "SECURITY.md",
        root / ".github" / "workflows" / "ci.yml",
    ]
    for path in required:
        assert path.is_file(), path
    readme = (root / "README.md").read_text()
    assert "reachy-hermes-agent" in readme
    assert "AGENT_BASE_URL" in readme
    assert "no-hardware" in readme


def test_no_markdown_trailing_whitespace() -> None:
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    _skip_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache", ".mypy_cache"}
    for path in root.rglob("*.md"):
        # Only lint our own docs — skip vendored/third-party trees (e.g. .venv site-packages).
        if _skip_dirs.intersection(path.parts):
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.rstrip() != line:
                offenders.append(f"{path.relative_to(root)}:{line_no}")
    assert offenders == []

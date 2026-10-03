"""Render a published changelog section plus versioned installation instructions."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPOSITORY = "https://github.com/YuhuanStudio/Yunshu"


def render_notes(changelog: str, version: str) -> str:
    """Fail closed on absent versions and derive the comparison from the older entry."""
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError(f"Invalid release version: {version}")
    headings = list(re.finditer(r"^## \[([^\]]+)\].*$", changelog, re.MULTILINE))
    for index, heading in enumerate(headings):
        if heading.group(1) == version:
            end = (
                headings[index + 1].start()
                if index + 1 < len(headings)
                else len(changelog)
            )
            body = changelog[heading.end() : end].strip()
            older = headings[index + 1].group(1) if index + 1 < len(headings) else None
            break
    else:
        raise ValueError(f"Missing changelog section: {version}")
    if not body:
        raise ValueError(f"Empty changelog section: {version}")
    # The renderer owns the footer, so copied draft compare links cannot survive.
    body = re.sub(r"^\[Full changelog[^\n]*\n?", "", body, flags=re.MULTILINE).rstrip()
    # GitHub release pages have no repository-relative Markdown base.
    body = re.sub(
        r"\]\((?![a-zA-Z][a-zA-Z0-9+.-]*:|#|/)([^)]+)\)",
        lambda match: f"]({REPOSITORY}/blob/v{version}/{match.group(1)})",
        body,
    )
    instructions = f"""### Install / upgrade

macOS with Apple Silicon and Python 3.13+. Choose your package manager:

```bash
# pip: run inside a virtual environment
python -m pip install --upgrade "yunshu[vision]=={version}"
# uv: install or replace the isolated CLI tool
uv tool install --upgrade --python 3.13 "yunshu[vision]=={version}"
# Homebrew: available after the tap formula is updated
brew update
brew install yuhuanstudio/tap/yunshu   # first install
brew upgrade yuhuanstudio/tap/yunshu  # existing install
yunshu --version
yunshu doctor
```

Homebrew can lag the release; check `yunshu --version`. For other modalities,
choose the extras listed in [the installation guide]({REPOSITORY}/blob/v{version}/README.md#quickstart).
"""
    compare = (
        f"{REPOSITORY}/compare/v{older}...v{version}"
        if older
        else f"{REPOSITORY}/commits/v{version}"
    )
    return f"{body}\n\n{instructions}\n[Full changelog]({compare})\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    parser.add_argument("--output", type=Path, default=Path("notes.md"))
    args = parser.parse_args()
    args.output.write_text(render_notes(args.changelog.read_text(), args.version))


if __name__ == "__main__":
    main()

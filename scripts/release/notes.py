"""Render a published changelog section plus versioned installation instructions."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPOSITORY = "https://github.com/YuhuanStudio/Yunshu"


def _drop_empty_subsections(body: str) -> str:
    """Remove ### headings with nothing under them (a header-only table counts as nothing)."""
    parts = re.split(r"(?m)^(?=### )", body)
    kept = []
    for part in parts:
        lines = part.strip().splitlines()
        if part.startswith("### "):
            rest = [ln for ln in lines[1:] if ln.strip()]
            rows = [ln for ln in rest if ln.startswith("|")]
            if not rest or (rows == rest and len(rows) <= 2):
                continue
        kept.append(part.strip())
    return "\n\n".join(k for k in kept if k)


def previous_dev(version: str) -> str | None:
    """The dev release before ``version`` (X.Y.Z.devN -> X.Y.Z.dev(N-1)); None for dev1 or a patch."""
    m = re.fullmatch(r"([0-9]+\.[0-9]+\.[0-9]+)\.dev([0-9]+)", version)
    if not m or int(m.group(2)) <= 1:
        return None
    return f"{m.group(1)}.dev{int(m.group(2)) - 1}"


def _section_body(changelog: str, section: str) -> str:
    headings = list(re.finditer(r"^## \[([^\]]+)\].*$", changelog, re.MULTILINE))
    for index, heading in enumerate(headings):
        if heading.group(1) == section:
            end = (
                headings[index + 1].start()
                if index + 1 < len(headings)
                else len(changelog)
            )
            return changelog[heading.end() : end].strip()
    return ""


def _entries(lines: list[str]) -> list[list[str]]:
    """Group a subsection's lines into entries: a "- " bullet with its indented continuation
    lines, or one table row; anything else is its own entry."""
    groups: list[list[str]] = []
    for line in lines:
        if (
            line.startswith((" ", "\t"))
            and line.strip()
            and groups
            and groups[-1][0].lstrip().startswith("- ")
        ):
            groups[-1].append(line)
        elif line.strip():
            groups.append([line])
    return groups


def only_new_entries(body: str, previous_body: str) -> str:
    """Keep the entries of ``body`` that ``previous_body`` did not already list: a dev release
    describes what changed since the previous dev release, not everything since the last patch
    (user 2026-11-12: dev1 and dev2 notes were mixed). Subsection headings stay; an entry is a
    "- " bullet (with its continuation lines) or a table row; a table keeps its header rows when
    one of its rows is new."""
    seen = {
        " ".join(x.strip() for x in group)
        for chunk in re.split(r"(?m)^(?=### )", previous_body)
        for group in _entries(chunk.splitlines()[1:])
    }
    out: list[str] = []
    for chunk in re.split(r"(?m)^(?=### )", body):
        lines = chunk.splitlines()
        if not lines or not lines[0].startswith("### "):
            continue  # the intro paragraph is replaced by the caller
        kept: list[str] = []
        table_head: list[str] = []
        for group in _entries(lines[1:]):
            text = group[0].strip()
            key = " ".join(x.strip() for x in group)
            if text.startswith("|"):
                if (
                    re.fullmatch(r"\|[\s:|-]+\|", text)
                    or not table_head
                    and not kept_table_row(kept)
                ):
                    table_head.append(group[0])
                    continue
                if key not in seen:
                    kept.extend(table_head)
                    table_head = []
                    kept.append(group[0])
            elif text.startswith("- ") and key not in seen:
                kept.extend(group)
        if kept:
            out.append("\n".join([lines[0], ""] + kept))
    return "\n\n".join(out)


def kept_table_row(kept: list[str]) -> bool:
    return any(line.strip().startswith("|") for line in kept)


def render_notes(
    changelog: str, version: str, previous_changelog: str | None = None
) -> str:
    """Fail closed on absent versions and derive the comparison from the older entry."""
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(\.dev[0-9]+)?", version):
        raise ValueError(f"Invalid release version: {version}")
    # a dev release (every push between patch releases, user 2026-11-10) carries [Unreleased]
    section = "Unreleased" if ".dev" in version else version
    headings = list(re.finditer(r"^## \[([^\]]+)\].*$", changelog, re.MULTILINE))
    for index, heading in enumerate(headings):
        if heading.group(1) == section:
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
    body = _drop_empty_subsections(body)
    since = previous_dev(version)
    if since and previous_changelog is not None:
        body = (
            f"Changes since {since}. The next patch release collects every dev release.\n\n"
            + only_new_entries(body, _section_body(previous_changelog, "Unreleased"))
        )
        older = since
    # 2026-11-11: v0.1.8.dev1 shipped empty headings and three entries; a release needs entries
    if not re.search(r"^- ", body, re.MULTILINE):
        raise ValueError(f"Changelog section {section} has no entries")
    # GitHub release pages have no repository-relative Markdown base.
    body = re.sub(
        r"\]\((?![a-zA-Z][a-zA-Z0-9+.-]*:|#|/)([^)]+)\)",
        lambda match: f"]({REPOSITORY}/blob/v{version}/{match.group(1)})",
        body,
    )
    dev = ".dev" in version
    brew = (
        ""
        if dev
        else "# Homebrew: available after the tap formula is updated\nbrew update\n"
        "brew install yuhuanstudio/tap/yunshu   # first install\n"
        "brew upgrade yuhuanstudio/tap/yunshu  # existing install\n"
    )
    brew_note = (
        "A pre-release: Homebrew stays on the latest stable version. "
        if dev
        else "Homebrew can lag the release; check `yunshu --version`. "
    )
    instructions = f"""### Install / upgrade

macOS with Apple Silicon and Python 3.13+. Choose your package manager:

```bash
# pip: run inside a virtual environment
python -m pip install --upgrade "yunshu[vision]=={version}"
# uv: install or replace the isolated CLI tool
uv tool install --upgrade --python 3.13 "yunshu[vision]=={version}"
{brew}yunshu --version
yunshu doctor
```

{brew_note}For other modalities,
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
    parser.add_argument(
        "--previous-changelog",
        type=Path,
        help="CHANGELOG.md at the previous dev tag: a devN (N > 1) lists only what is new since it",
    )
    args = parser.parse_args()
    previous = args.previous_changelog.read_text() if args.previous_changelog else None
    args.output.write_text(
        render_notes(args.changelog.read_text(), args.version, previous)
    )


if __name__ == "__main__":
    main()

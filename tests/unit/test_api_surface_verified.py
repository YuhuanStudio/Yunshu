"""docs/guides/API_SURFACE.md: every route / parameter row says how it is verified, and no row says
"real" without a date (a real-server claim must be dated evidence, not a general statement)."""

from __future__ import annotations

import re
from pathlib import Path

DOC = Path(__file__).resolve().parents[2] / "docs/guides/API_SURFACE.md"
SECTIONS = (
    "## OpenAI",
    "### Chat completions: parameters and fields",
    "## Anthropic",
    "## Ollama-compatible (`/api/*`)",
    "## Tokenizer (vLLM schema)",
    "## Retrieval (vLLM-style)",
    "## Other",
    "## Yunshu extensions",
)
OK = re.compile(
    r"^(real \d{4}-\d{2}-\d{2}\b|unit\b|audit\b|not applicable|not offered|not registered)"
)


def rows():
    section = ""
    for ln in DOC.read_text().splitlines():
        if ln.startswith(("## ", "### ")):
            section = ln
        if section in SECTIONS and ln.startswith("| "):
            yield section, [c.strip() for c in ln.strip().strip("|").split("|")]


def test_every_row_has_a_verified_cell():
    seen = 0
    for section, cells in rows():
        if set("".join(cells)) <= set("-: "):
            continue
        if cells[0] in ("Route", "Item", "Route / field"):
            assert cells[-1] == "Verified", f"{section}: header without Verified"
            continue
        seen += 1
        assert OK.match(cells[-1]), (
            f"{section} :: {cells[0][:60]} -> {cells[-1][:80]!r}"
        )
    assert seen > 60


def test_real_claims_are_dated_and_name_the_check():
    for _section, cells in rows():
        v = cells[-1]
        if v.startswith("real"):
            assert re.match(r"real \d{4}-\d{2}-\d{2}: ", v), v[:60]
            assert any(
                k in v for k in ("`routes`", "`wire`", "absent-capability", "real")
            ), v[:80]

"""Per-request evidence lines tolerate fields the engine adds (tool calls, grammar, thinking, images, cached/dedup).

wire-1108rest, Splash p3, kind/tool-chat: the done line carries 'tools 2·c5b1f9df' and the per-row regex, written for
'input · cached [· deduplicated] · output · TTFT · tok/s · tok/cycle', rejected it, as it had rejected 'deduplicated'
before. The class is a fixed field order: the patterns now require only the fields the verdict needs (Splash: input,
output, tok/cycle; key=value engines: each key anywhere on the line).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
import fullcmp_engaged as fe  # noqa: E402

# the real wire-1108rest Splash done line of the kind/tool-chat cell
SPLASH_TOOLS = "Done · input 8,573 · cached 0 · output 27 · tools 2·c5b1f9df · TTFT 7.0s · 135.8 tok/s · 5.2 tok/cycle"


def _item(engine, mode, nth=-1):
    table = fe.load_table()
    items = [i for i in table["engines"][engine]["modes"][mode]["evidence"] if i["kind"] == "acceptance_regex" and i.get("per_request")]
    return items[nth]


def _rows(*pts):
    return [{"pt": p, "ct": 16, "cached": 0} for p in pts]


def test_splash_done_line_with_a_tools_segment_is_per_row_evidence():
    rx = re.compile(_item("splash-1.5.0", "dflash")["pattern_or_path"])
    m = rx.search(SPLASH_TOOLS)
    assert m and m.group("prompt") == "8,573" and m.group("tpc") == "5.2"


@pytest.mark.parametrize(
    "line,prompt,tpc",
    [
        ("Done · input 15 · cached 0 · output 10 · TTFT 0.2s · 118.5 tok/s · 4.5 tok/cycle", "15", "4.5"),
        ("Done · input 1,024 · cached 992 · deduplicated 64 · output 64 · TTFT 0.1s · 67.5 tok/s · 2.6 tok/cycle", "1,024", "2.6"),
        (SPLASH_TOOLS, "8,573", "5.2"),
        ("Done · input 900 · cached 0 · output 40 · grammar json-schema · TTFT 1.0s · 90.0 tok/s · 3.1 tok/cycle", "900", "3.1"),
        ("Done · input 900 · images 1 · cached 0 · thinking 120 · output 40 · TTFT 1.0s · 90.0 tok/s · 3.1 tok/cycle", "900", "3.1"),
        ("Done · input 7 · output 3 · 2.0 tok/cycle", "7", "2.0"),  # only the required segments
    ],
)
def test_splash_pattern_requires_only_input_output_and_tok_per_cycle(line, prompt, tpc):
    m = re.search(_item("splash-1.5.0", "dflash")["pattern_or_path"], line)
    assert m and m.group("prompt") == prompt and m.group("tpc") == tpc


@pytest.mark.parametrize("line", ["Done · output 10 · 2.0 tok/cycle", "Done · input 10 · 2.0 tok/cycle", "Done · input 10 · output 3 · TTFT 1s"])
def test_splash_pattern_still_rejects_a_line_missing_a_required_segment(line):
    assert not re.search(_item("splash-1.5.0", "dflash")["pattern_or_path"], line)


def test_the_real_tool_chat_cell_correlates_end_to_end_through_the_verdict():
    """Failing-before: the old pattern found no done line for the 8,573-token row ('no done line': startup_only)."""
    ready = [
        "00:00:01 Done · input 15 · cached 0 · output 10 · TTFT 0.2s · 118.5 tok/s · 4.5 tok/cycle",
        "00:00:02 Done · input 15 · cached 0 · output 10 · TTFT 0.1s · 120.1 tok/s · 4.5 tok/cycle",
    ]
    log = "Decode cycles verify every row of the draft's block (--verify-rows auto)\n" + "\n".join([*ready, SPLASH_TOOLS]) + "\n"
    ev = fe.evaluate("splash-1.5.0", "dflash", log, _rows(8573), fe.load_table(), ready_offset=0)
    assert ev["engaged_mode"] == "dflash", ev


# --- the other engines' per-request patterns, with an extra field the engine could add ----------------------------


def _perturb(engine, line):
    if engine.startswith("kapitsa"):
        return line.replace(", KFlash (DFlash2)", ", tools 2, KFlash (DFlash2)").replace(", stop,", ", stop, grammar json,")
    if engine.startswith("omlx"):
        return line.replace(" tok/cycle=", " newfield=7 tok/cycle=").replace(" accept=", " extra=1 accept=")
    if engine.startswith("mlx-serve"):
        return line.replace(" accepts=", " newfield=3 accepts=").replace(" drafted=", " zz=1 drafted=")
    if engine.startswith("tf-"):
        return line.replace(" cached=", " tools=2 grammar=json cached=")
    return line


@pytest.mark.parametrize(
    "engine,mode",
    [("kapitsa-0.2.26", "dflash"), ("omlx-0.7.3", "dflash"), ("omlx-0.7.3rc1", "dflash"), ("mlx-serve-26.11.1", "mtp"), ("tf-1.2.1", "dflash"), ("tf-1.1.8", "dflash")],
)
def test_per_request_patterns_survive_an_extra_field(engine, mode):
    item = _item(engine, mode)
    example = item["real_example"]
    rx = re.compile(item["pattern_or_path"])
    base = rx.search(example)
    assert base, "the real example must match"
    grown = _perturb(engine, example)
    assert grown != example
    m = rx.search(grown)
    assert m, f"{engine}: an extra field broke the pattern"
    for g in base.groupdict():
        if base.group(g) is not None:
            assert m.group(g) == base.group(g), (engine, g)


# the real Splash kind cells of wire-1108rest (done lines copied from their server logs): two readiness pings, then
# the cell's one request; rows' prompt counts are the usage values the cell recorded
SPLASH_KIND_CELLS = {
    "cjk": ("Done · input 8,211 · cached 0 · output 512 · TTFT 6.7s · 59.1 tok/s · 2.3 tok/cycle", 8211),
    "json-schema": ("Done · input 8,216 · cached 0 · output 263 · TTFT 6.7s · 156.3 tok/s · 6.1 tok/cycle", 8216),
    "tool-chat": (SPLASH_TOOLS, 8573),
}


@pytest.mark.parametrize("kind", sorted(SPLASH_KIND_CELLS))
def test_each_real_splash_kind_cell_verdict_is_dflash(kind):
    line, pt = SPLASH_KIND_CELLS[kind]
    ready = [
        "13:44:42 Done · input 15 · cached 0 · output 10 · TTFT 0.1s · 124.4 tok/s · 4.5 tok/cycle",
        "13:44:43 Done · input 15 · cached 0 · output 10 · TTFT 0.0s · 124.5 tok/s · 4.5 tok/cycle",
    ]
    log = "Decode cycles verify every row of the draft's block (--verify-rows auto)\n" + "\n".join([*ready, "13:44:58 " + line]) + "\n"
    ev = fe.evaluate("splash-1.5.0", "dflash", log, _rows(pt), fe.load_table(), ready_offset=0)
    assert ev["engaged_mode"] == "dflash", (kind, ev)

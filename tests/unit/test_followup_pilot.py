"""CPU unit tests for the follow-up pilot's pure helpers (no server, no GPU)."""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "research" / "followup_pilot.py"
_spec = importlib.util.spec_from_file_location("followup_pilot", _PATH)
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)

LINE = (
    "2026-11-12 12:24:59,953 INFO yunshu_engine.hit_stages: turn stages: ids=33567 hit=32768 | "
    "standing=0.0 vocab=0.0 add=21.4 (lookup=19.3 plan=0.8 boundaries=0.4 hash=0.4 window=0.0) "
    "wait_first=2846.8 emit=0.0 other=0.7 total=2868.9 ms"
)


def test_build_followup_resends_reply_verbatim():
    msgs = pilot.build_followup("PROMPT", "REPLY\n")
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"] == "REPLY\n"  # the trailing newline is sent back as produced
    assert msgs[2]["content"] == pilot.FOLLOW_TAIL


def test_parse_turn_stages_reads_every_field():
    rows = pilot.parse_turn_stages(LINE + "\nnoise\n" + LINE.replace("ids=33567", "ids=5"))
    assert len(rows) == 2
    assert rows[0]["ids"] == 33567 and rows[0]["hit"] == 32768
    assert rows[0]["lookup"] == 19.3 and rows[0]["wait_first"] == 2846.8
    assert rows[1]["ids"] == 5


def test_parse_turn_stages_empty_when_absent():
    assert pilot.parse_turn_stages("no stage lines here") == []


def test_cached_tokens_reads_openai_usage_shape():
    assert pilot.cached_tokens({"prompt_tokens_details": {"cached_tokens": 32768}}) == 32768
    assert pilot.cached_tokens({}) == 0
    assert pilot.cached_tokens(None) == 0

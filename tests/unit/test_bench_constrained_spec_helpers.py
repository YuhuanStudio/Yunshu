"""Harness helpers: the token-receipt race and the second-turn builder."""

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest

_DIR = Path(__file__).resolve().parents[2] / "scripts/research"


def _load():
    sys.path.insert(0, str(_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            "cspec_bench", _DIR / "bench_constrained_spec.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(_DIR))


def test_read_trace_waits_for_a_writer_that_finishes_after_the_stream(tmp_path):
    bench = _load()
    trace = tmp_path / "t.jsonl"

    def writer():
        time.sleep(0.3)
        trace.write_text(json.dumps({"a": 1}) + "\n")

    threading.Thread(target=writer).start()
    # Before the fix the harness read the file immediately and hit
    # JSONDecodeError("Expecting value") on the not-yet-written receipt.
    assert bench.read_trace(trace, 1) == [{"a": 1}]


def test_read_trace_fails_closed_when_the_record_never_arrives(tmp_path):
    bench = _load()
    with pytest.raises(RuntimeError):
        bench.read_trace(tmp_path / "missing.jsonl", 1, timeout=0.2)


def test_second_turn_extends_the_first_prefix_for_text_and_tools():
    bench = _load()
    text = bench.second_turn("q", ["answer", "", [], "stop"], {})
    assert text[:2] == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "answer"},
    ]
    tool = bench.second_turn(
        "q", ["", "", [{"name": "f", "arguments": "{}"}], "tool_calls"], {}
    )
    assert [m["role"] for m in tool] == ["user", "assistant", "tool", "user"]
    assert tool[1]["tool_calls"][0]["function"]["name"] == "f"


def test_read_trace_ignores_an_unfinished_next_line(tmp_path):
    bench = _load()
    trace = tmp_path / "partial.jsonl"
    trace.write_text('{"n": 1}\n{"n":')
    assert bench.read_trace(trace, 1, timeout=0.2) == [{"n": 1}]


def test_trace_cursor_advances_across_cases_and_phases(tmp_path):
    bench = _load()
    trace = tmp_path / "tokens.jsonl"
    trace.write_text("".join(json.dumps({"token_ids": [n]}) + "\n" for n in range(6)))
    cursor = bench.TraceCursor(trace)
    assert [
        cursor.next()["token_ids"] for _case in range(2) for _phase in range(3)
    ] == [[n] for n in range(6)]


def test_agent_title_is_greedy_but_primary_bodies_are_unchanged(monkeypatch):
    bench = _load()
    seen = []
    monkeypatch.setattr(
        bench.tfbench, "send", lambda url, body, *a, **k: seen.append(body)
    )
    original = bench.tfbench.send

    def agent(srv, out, args):
        bench.tfbench.send("url", {"temperature": 0.7, "seed": 1234})
        bench.tfbench.send("url", {"temperature": 0.7, "tools": [{}], "seed": 1234})

    monkeypatch.setattr(bench.tfbench, "part_agent", agent)
    bench.run_agent(None, None, None)
    assert seen == [
        {"temperature": 0, "seed": 1234},
        {"temperature": 0.7, "tools": [{}], "seed": 1234},
    ]
    assert bench.tfbench.send is original


@pytest.mark.parametrize("phase", ["warm", "turn2"])
def test_cache_reuse_probe_fails_closed_on_a_miss(phase):
    bench = _load()
    with pytest.raises(RuntimeError, match="did not reuse"):
        bench.require_cache_reuse({"xy": {"cached_tokens": 0}}, phase)
    bench.require_cache_reuse({"xy": {"cached_tokens": 29}}, phase)

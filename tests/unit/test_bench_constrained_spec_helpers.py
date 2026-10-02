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

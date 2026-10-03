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


def test_agent_title_is_greedy_but_primary_bodies_are_unchanged(monkeypatch, capsys):
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
    assert capsys.readouterr().out.splitlines() == [
        "agent request finished",
        "agent request finished",
    ]


@pytest.mark.parametrize("phase", ["warm", "turn2"])
def test_cache_reuse_probe_fails_closed_on_a_miss(phase):
    bench = _load()
    with pytest.raises(RuntimeError, match="did not reuse"):
        bench.require_cache_reuse({"xy": {"cached_tokens": 0}}, phase)
    bench.require_cache_reuse({"xy": {"cached_tokens": 29}}, phase)


def test_cache_diff_identifies_state_slot_and_ignores_container_class():
    bench = _load()
    ar = [
        [3, "KVCache", [1, 4, 185, 256], "bf16", "a"],
        [3, "KVCache", [1, 4, 185, 256], "bf16", "b"],
    ]
    spec = [
        [3, "BatchKVCache", [1, 4, 185, 256], "bf16", "a"],
        [3, "BatchKVCache", [1, 4, 185, 256], "bf16", "c"],
    ]
    assert bench.cache_state_differences(ar, spec) == [
        {"layer": 3, "slot": 1, "baseline": ar[1], "speculative": spec[1]}
    ]
    assert bench.cache_state_differences(ar, spec[:1]) == [
        {"layer": 3, "slot": 1, "baseline": ar[1], "speculative": None}
    ]


def test_structured_receipt_distinguishes_truncation_from_valid_schema():
    from jsonschema.exceptions import ValidationError

    bench = _load()
    extra = {
        "response_format": {
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {"n": {"type": "integer"}},
                    "required": ["n"],
                    "additionalProperties": False,
                }
            }
        }
    }
    assert bench.structured_receipt(["{", "", [], "length"], extra) == {
        "structured_complete": False,
        "structured_status": "length_limit",
    }
    assert bench.structured_receipt(['{"n": 1}', "", [], "stop"], extra) == {
        "structured_complete": True,
        "structured_status": "schema_valid",
    }
    with pytest.raises(ValidationError):
        bench.structured_receipt(['{"n": "wrong"}', "", [], "stop"], extra)


def test_required_tool_receipt_checks_arguments_and_missing_calls():
    bench = _load()
    extra = {
        "tool_choice": "required",
        "tools": [
            {
                "function": {
                    "name": "f",
                    "parameters": {"type": "object", "required": ["x"]},
                }
            }
        ],
    }
    assert bench.structured_receipt(
        ["", "", [{"name": "f", "arguments": '{"x": 1}'}], "tool_calls"], extra
    )["structured_complete"]
    with pytest.raises(ValueError, match="required tool"):
        bench.structured_receipt(["", "", [], "stop"], extra)


def test_agent_trace_waits_for_primary_and_auxiliary_receipts(tmp_path):
    bench = _load()
    trace = tmp_path / "agent.jsonl"
    rows = [{"part": "agent", "i": i} for _body in range(2) for i in range(3)]
    trace.write_text("".join(json.dumps({"n": n}) + "\n" for n in range(7)))

    def writer():
        time.sleep(0.3)
        with trace.open("a") as out:
            out.write('{"n": 7}\n')

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        assert bench.read_agent_trace(trace, rows) == [{"n": n} for n in range(8)]
    finally:
        thread.join()
    with pytest.raises(RuntimeError, match="no primary"):
        bench.read_agent_trace(trace, [{"complete": True}])


def test_http_receipt_rejects_nonfinite_json_logprobs(monkeypatch):
    from types import SimpleNamespace

    bench = _load()
    chunk = {
        "choices": [
            {
                "delta": {"content": "x"},
                "finish_reason": "stop",
                "logprobs": {
                    "content": [
                        {
                            "token": "x",
                            "logprob": -1.0,
                            "top_logprobs": [
                                {"token": "masked", "logprob": float("-inf")}
                            ],
                        }
                    ]
                },
            }
        ],
        "usage": {"completion_tokens": 1, "prompt_tokens": 1},
    }

    class Response:
        def __enter__(self):
            return iter([(f"data: {json.dumps(chunk)}\n").encode(), b"data: [DONE]\n"])

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(bench.urllib.request, "urlopen", lambda *a, **kw: Response())
    with pytest.raises(ValueError, match="nonfinite"):
        bench.send(SimpleNamespace(url="http://localhost", model="test"), "q", {}, 1)

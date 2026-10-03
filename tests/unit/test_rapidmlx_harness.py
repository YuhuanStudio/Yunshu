"""Comparison receipts reject incomplete streams and malformed agent calls."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts/research/rapidmlx" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shapes = load("agent_shapes")
harness = load("head_to_head")


def response(name, arguments):
    if name.startswith("claude-"):
        return {"content": [{"type": "tool_use", "name": "shell", "input": arguments}]}
    if name.startswith("codex-"):
        return {
            "output": [
                {
                    "type": "function_call",
                    "name": "shell",
                    "arguments": json.dumps(arguments),
                }
            ]
        }
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "shell",
                                "arguments": json.dumps(arguments),
                            }
                        }
                    ]
                }
            }
        ]
    }


@pytest.mark.parametrize("name", ["openai-auto", "claude-forced", "codex-auto"])
def test_agent_grade_checks_exact_schema(name):
    good = {"command": "printf rapidmlx", "timeout": 1000}
    assert shapes.grade(name, response(name, good))[0]
    # JSON Schema integer includes integral JSON numbers such as 1000.0.
    assert shapes.grade(name, response(name, dict(good, timeout=1000.0)))[0]
    for malformed in [
        dict(good, timeout="1000"),
        dict(good, timeout=True),
        dict(good, extra="unexpected"),
        dict(good, command="rm -rf something"),
    ]:
        assert not shapes.grade(name, response(name, malformed))[0]
    assert not shapes.grade(name, {})[0]


def sse(done=True, usage=True):
    events = [
        {"choices": [{"delta": {"content": "hello"}}]},
        {"choices": [{"delta": {"content": " world"}, "finish_reason": "length"}]},
    ]
    if usage:
        events.append({"usage": {"completion_tokens": 2, "prompt_tokens": 4}})
    data = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
    return data + (b"data: [DONE]\n\n" if done else b"")


def test_sse_receipt_preserves_raw_and_usage(monkeypatch, tmp_path):
    raw = sse()
    monkeypatch.setattr(
        harness.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(raw)
    )
    times = iter([10.0, 10.2, 10.5, 10.6])
    monkeypatch.setattr(harness.time, "perf_counter", lambda: next(times))
    row = harness.request("http://localhost:18998", {}, tmp_path / "raw.sse")
    assert row["output"] == "hello world"
    assert row["ttft_s"] == pytest.approx(0.2)
    assert row["decode_tps"] == pytest.approx(1 / 0.3)
    assert row["done"]
    assert (tmp_path / "raw.sse").read_bytes() == raw


@pytest.mark.parametrize("done,usage", [(False, True), (True, False)])
def test_incomplete_stream_is_not_success(monkeypatch, done, usage):
    monkeypatch.setattr(
        harness.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(sse(done, usage)),
    )
    with pytest.raises(RuntimeError, match="incomplete SSE"):
        harness.request("http://localhost:18998", {})


def test_stream_error_is_not_silently_a_timing(monkeypatch):
    monkeypatch.setattr(
        harness.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(b'data: {"error":{"message":"failed"}}\n\n'),
    )
    with pytest.raises(RuntimeError, match="failed"):
        harness.request("http://localhost:18998", {})


def test_all_agent_shapes_include_automatic_and_forced_choice():
    cases = list(shapes.cases("model"))
    assert len(cases) == 6
    assert {route.split("?")[0] for _, route, _ in cases} == {
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/responses",
    }
    assert all(body["model"] == "model" for _, _, body in cases)


def test_different_result_files_never_overwrite_raw_arm_artifacts(tmp_path):
    first = harness.arm_directory(tmp_path / "tiny.jsonl", "rapid", 0, 128)
    second = harness.arm_directory(tmp_path / "27b-smoke.jsonl", "rapid", 0, 128)
    assert first != second
    assert first == tmp_path / "tiny" / "rapid-r0-n128"
    assert second == tmp_path / "27b-smoke" / "rapid-r0-n128"


matrix = load("matrix")


def test_resume_reuses_only_whole_successful_arms():
    cases = [
        "startup",
        "cold",
        "warm",
        "turn2",
        "concurrent8",
        "physical_memory",
        "engaged_mode",
        "memory",
    ]
    rows = [
        dict(profile="rapid-default", rep=0, size=1024, case=c, done=True)
        for c in cases
    ]
    assert len(matrix.completed_arms(rows)) == 1
    assert not matrix.completed_arms(rows[:-1])
    assert not matrix.completed_arms(rows, tool_eval=True)
    assert not matrix.completed_arms(rows + [dict(rows[0], error="interrupted")])
    assert not matrix.completed_arms(
        [dict(r, done=False) if r["case"] == "cold" else r for r in rows]
    )


def test_resume_does_not_reuse_failed_eval_receipts():
    cases = [
        "startup",
        "cold",
        "warm",
        "turn2",
        "concurrent8",
        "physical_memory",
        "engaged_mode",
        "memory",
        "rapid_tool_eval",
        "census_replay",
        "agent_shapes",
    ]
    rows = [
        dict(
            profile="yunshu-default",
            rep=0,
            size=1024,
            case=c,
            done=True,
            exists=True,
            rc=0,
        )
        for c in cases
    ]
    assert len(matrix.completed_arms(rows, True)) == 1
    assert not matrix.completed_arms(
        [dict(r, rc=1) if r["case"] == "census_replay" else r for r in rows], True
    )

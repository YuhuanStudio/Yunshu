"""CPU tests for the M3 sweep driver and its job checks (scripts/dev/m3sweep, m3sweep_jobs)."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))

import m3sweep_jobs as jobs  # noqa: E402

from .wire_clients import Out  # noqa: E402


def _driver():
    path = str(ROOT / "scripts/dev/m3sweep")
    loader = importlib.machinery.SourceFileLoader("m3sweep_driver", path)
    spec = importlib.util.spec_from_loader("m3sweep_driver", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _out(**kw):
    base = dict(finish="stop", prompt=5, completion=3, text="Hi")
    base.update(kw)
    return Out(**base)


def test_basic_ok_and_problems():
    assert jobs.check_case("basic", {"max_tokens": 32}, "chat", False, _out()) == []
    assert jobs.check_case("basic", {"max_tokens": 32}, "chat", False, _out(text=""))
    assert jobs.check_case(
        "basic", {"max_tokens": 4}, "chat", False, _out(completion=9)
    )
    assert jobs.check_case("basic", {}, "chat", False, _out(prompt=None))
    assert jobs.check_case("basic", {}, "chat", False, _out(finish="weird"))
    assert jobs.check_case("basic", {}, "chat", False, _out(cached=9))


def test_truncate_and_stop():
    assert (
        jobs.check_case(
            "truncate", {"max_tokens": 3}, "chat", False, _out(finish="length")
        )
        == []
    )
    assert jobs.check_case(
        "truncate", {"max_tokens": 3}, "chat", False, _out(finish="stop")
    )
    assert jobs.check_case("stop", {}, "chat", False, _out(text="hello"))
    assert (
        jobs.check_case("stop", {}, "chat", False, _out(text="hi", finish="length"))
        == []
    )


def test_tools_and_schema():
    call = [("get_weather", {"city": "Paris"})]
    ok = _out(finish="tool_calls", tools=call, text="")
    assert jobs.check_case("tool_required", {}, "chat", False, ok) == []
    assert jobs.check_case("tool_required", {}, "chat", False, _out())
    assert jobs.check_case(
        "tool_required", {}, "chat", False, _out(finish="stop", tools=call)
    )
    assert jobs.check_case("tool_none", {}, "chat", False, ok)
    assert jobs.check_case(
        "tool_serial", {}, "chat", False, _out(finish="tool_calls", tools=call * 2)
    )
    assert jobs.check_case("schema", {}, "chat", False, _out(text='{"a": 3}')) == []
    assert jobs.check_case("schema", {}, "chat", False, _out(text="nope"))


def test_stream_prompt_mismatch():
    assert jobs.compare_stream("basic", "chat", _out(prompt=5), _out(prompt=6))
    assert not jobs.compare_stream(
        "basic", "ollama_chat", _out(prompt=5), _out(prompt=6)
    )


def test_error_shapes():
    ok = {"error": {"message": "x", "type": "invalid_request_error"}}
    assert jobs.check_error("l", 400, ok, "openai") == []
    assert jobs.check_error("l", 200, ok, "openai")
    assert jobs.check_error("l", 400, {"detail": "x"}, "openai")
    anth = {"type": "error", "error": {"type": "invalid_request_error", "message": "m"}}
    assert jobs.check_error("l", 400, anth, "anthropic") == []
    assert jobs.check_error("l", 400, ok, "anthropic")


def test_pytest_summary_and_skips():
    assert jobs.parse_pytest_summary("x\n3 passed, 2 skipped in 1s") == (3, 0, 2)
    assert jobs.parse_pytest_summary("1 failed, 2 passed in 1s") == (2, 1, 0)
    assert jobs.parse_pytest_summary("no tests ran") is None
    txt = "SKIPPED [1] a.py:3: /m/Qwen2.5-3B-Instruct-4bit not available\nSKIPPED [1] b.py: other"
    assert len(jobs.skipped_models(txt, ["/x/Qwen2.5-3B-Instruct-4bit"])) == 1


def test_plan_and_verdict(tmp_path):
    d = _driver()
    js = d.plan("abcdef1234", tmp_path)
    assert {j["name"] for j in js} >= {"wire-q25-3b", "agent-q25-3b", "units"}
    for j in js:
        assert j["submit"][:2] == ["--device", "m3"]
        assert j["label"].startswith("m3lane-")
        assert float(j["submit"][3]) <= 28
    assert all(j["name"].startswith("wire") for j in d.plan("a", tmp_path, {"wire"}))
    assert "wire-27b" in {j["name"] for j in d.plan("a", tmp_path, None, True)}
    # fail closed: nothing written -> FAIL; complete+pass -> PASS; failed/incomplete/m5 -> FAIL
    one = [js[0]]
    assert d.verdict(one, 0)["verdict"] == "FAIL"
    out = one[0]["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"complete": True, "pass": True}))
    assert d.verdict(one, 0)["verdict"] == "PASS"
    assert d.verdict(one, 1)["verdict"] == "FAIL"
    out.write_text(json.dumps({"complete": True, "pass": True, "device": "m5"}))
    assert d.verdict(one, 0)["verdict"] == "FAIL"
    out.write_text(json.dumps({"complete": False, "pass": True}))
    assert d.verdict(one, 0)["verdict"] == "FAIL"
    out.write_text(json.dumps({"complete": True, "pass": False, "failures": ["x"]}))
    v = d.verdict(one, 0)
    assert v["verdict"] == "FAIL" and "x" in v["jobs"][one[0]["name"]]["problems"]
    assert d.verdict([], 0)["verdict"] == "FAIL"


def test_structural_session_judge():
    def row(k, cached=0, calls=(), text="", ended=True, inp=1000):
        return {
            "kind": "req",
            "req": k,
            "ended": ended,
            "text": text,
            "tool_calls": list(calls),
            "usage": {"input_tokens": inp, "cache_read_input_tokens": cached},
        }

    good = [row(1), row(2, cached=1000), row(3, cached=1000), row(4, cached=1000)]
    assert jobs.structural_session_problems(good, 3) == []
    assert jobs.structural_session_problems(good[:2], 3)
    assert jobs.structural_session_problems([*good[:3], row(4, text="<tool_call>")], 3)
    assert jobs.structural_session_problems([*good[:3], row(4, cached=10)], 3)
    bad_call = {"name": "Read", "input": "oops"}
    assert jobs.structural_session_problems([*good[:3], row(4, calls=[bad_call])], 3)


def test_lenient_conc_and_lock():
    out = "JUDGE FAIL: long/solo: wrong answer\nJUDGE FAIL: c2/short: differs from solo: a vs b\n"
    assert jobs.lenient_conc_problems(out, 1) == ["c2/short: differs from solo: a vs b"]
    assert jobs.lenient_conc_problems("RESULT PASS", 0) == []
    assert jobs.lenient_conc_problems("crash", 2)
    lock = '[[package]]\nname = "Open_AI"\nversion = "3.0"\n[[package]]\nname = "x"\nversion = "1"\n'
    locked = jobs.locked_versions(lock)
    assert locked == {"open-ai": "3.0", "x": "1"}
    assert jobs.lock_mismatches(locked, {"open-ai": "2.0", "x": "1", "y": "9"}) == {
        "open-ai": ("3.0", "2.0")
    }

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
    assert all(
        j["name"].startswith(("wire", "env")) for j in d.plan("a", tmp_path, {"wire"})
    )
    allowed = {
        "Qwen3.5-0.8B-MLX-bf16",
        "Qwen3.5-2B-MLX-bf16",
        "Qwen3.5-9B-MLX-4bit",
        "Qwen2.5-3B-Instruct-4bit",
        "gemma-4-e2b-it-4bit",
        "Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
        "Qwen3-ASR-1.7B-bf16",
    }
    for j in d.plan("a", tmp_path):
        used = {
            Path(c).name for c in j["cmd"] if c.startswith("/Volumes/P5Plus/models/")
        }
        assert used <= allowed, used
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

    good = [row(1), row(2, cached=1000), row(3, cached=2000), row(4, cached=3000)]
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


def test_every_job_argv_parses(tmp_path):
    d = _driver()
    for j in d.plan("abc1234", tmp_path):
        argv = j["cmd"]
        assert argv[1] == d.JOBS_PY
        ns = jobs.build_parser().parse_args(argv[2:])
        assert ns.cmd in (
            "env",
            "wire",
            "agent",
            "units",
            "routes",
            "omni",
        ) and ns.out == str(j["out"])
        assert (tmp_path / "x").parent == tmp_path  # plan is pure: nothing created
        assert all(
            Path(m).is_absolute()
            for m in (ns.model if isinstance(ns.model, list) else [ns.model])
            if m
        )


def test_stop_sequence_is_stop_and_no_excused_failures():
    assert (
        jobs.check_case(
            "stop", {}, "messages", False, _out(finish="stop_sequence", text="hi")
        )
        == []
    )
    # a reply that reasoned is not checked for stop strings (they apply to the answer only)
    assert (
        jobs.check_case(
            "stop", {}, "completions", False, _out(text="hello", reasoning=2)
        )
        == []
    )
    assert jobs.check_case("stop", {}, "completions", False, _out(text="hello"))
    assert jobs.KNOWN_GAPS == {}
    assert (
        jobs.classify("tool_named/chat/json: forced tool call missing", False) is None
    )


def _m3_models():
    sys.path.insert(0, str(ROOT / "scripts/dev"))
    import gpuq_remote

    return gpuq_remote.M3_MODELS


def test_routes_job_in_plan_with_two_models_and_a_multi_server(tmp_path):
    d = _driver()
    plan = {j["name"]: j for j in d.plan("abc1234", tmp_path)}
    cmd = plan["routes"]["cmd"]
    # both small models one after the other (three ports in the lane), then the multi server
    assert cmd.count("--model") == 2 and cmd.count("--multi") == 2
    names = [
        j["name"] for j in d.plan("abc1234", tmp_path, {"routes"}) if j["name"] != "env"
    ]
    assert names == [
        "routes",
        "routes-speech",
        "routes-ocr",
        "routes-image",
        "routes-embed",
        "routes-embed2",
    ]
    # one server per modality, every checkpoint on the M3 allowlist, declared memory under the cap
    allowed = d.MODELS and _m3_models()
    for j in d.plan("abc1234", tmp_path):
        if j["name"].startswith("routes-"):
            mem = int(j["submit"][j["submit"].index("--mem-gb") + 1])
            tmo = int(j["submit"][j["submit"].index("--timeout") + 1])
            assert mem <= 28 and tmo <= 20, j["name"]
            for a in j["cmd"]:
                if "=/Volumes/" in a:
                    assert a.rsplit("/", 1)[1] in allowed, a


def test_route_coverage_verdict_fails_closed(tmp_path):
    import route_checks as rc

    d = _driver()
    routes_jobs = [
        j for j in d.plan("abc1234", tmp_path, {"routes"}) if j["name"] == "routes"
    ]
    every = sorted(rc.served_routes())

    def out(verified, ok=True):
        routes_jobs[0]["out"].write_text(
            json.dumps(
                {"complete": True, "pass": ok, "verified_served_routes": verified}
            )
        )

    out(every[:-1])
    assert d.route_coverage(routes_jobs) == [
        f"route has no passing SERVED check: {every[-1]}"
    ]
    v = d.verdict(routes_jobs, 0)
    assert v["verdict"] == "FAIL" and "route-coverage" in v["jobs"]
    out(every)
    assert d.route_coverage(routes_jobs) == []
    assert d.verdict(routes_jobs, 0)["verdict"] == "PASS"
    assert d.route_coverage([]) == []  # a sweep without the routes job says nothing


def test_run_route_checks_records_pass_fail_skip(monkeypatch, tmp_path):
    import types

    import route_checks as rc

    def ok(c):
        pass

    def bad(c):
        rc.expect(False, "boom")

    def sk(c):
        rc.skip("no vision")

    monkeypatch.setattr(
        rc,
        "REGISTRY",
        {
            "a": rc.Check("a", ok, ("GET /a",)),
            "b": rc.Check("b", bad, ("GET /b",)),
            "c": rc.Check("c", sk, ("GET /c",)),
            "m": rc.Check("m", ok, ("GET /m",), "multi"),
        },
    )
    srv = types.SimpleNamespace(
        proc=types.SimpleNamespace(poll=lambda: None, returncode=None)
    )
    res = {"checks": {}, "failures": [], "_out": str(tmp_path / "o.json")}
    jobs.run_route_checks(object(), "main", None, res, srv)
    assert {k: v["status"] for k, v in res["checks"].items()} == {
        "a": "pass",
        "b": "fail",
        "c": "skip",
    }
    assert res["failures"] and "boom" in res["failures"][0]
    dead = types.SimpleNamespace(
        proc=types.SimpleNamespace(poll=lambda: 1, returncode=1)
    )
    res2 = {"checks": {}, "failures": [], "_out": str(tmp_path / "o2.json")}
    jobs.run_route_checks(object(), "main", None, res2, dead)
    assert all(v["status"] == "fail" for v in res2["checks"].values())  # fail closed


def test_error_shape_helpers():
    import route_checks as rc

    class R:
        def __init__(self, status, body):
            self.status_code, self._b = status, body
            self.text = json.dumps(body)

        def json(self):
            return self._b

    rc.err_ok(R(404, {"error": {"message": "m", "type": "t"}}), "openai")
    rc.err_ok(
        R(
            400,
            {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "m"},
            },
        ),
        "anthropic",
    )
    rc.err_ok(R(400, {"error": "msg"}), "ollama")
    for fam, r in [
        ("openai", R(404, {"detail": "x"})),
        ("openai", R(200, {"error": {"message": "m", "type": "t"}})),
        ("anthropic", R(400, {"error": {"message": "m", "type": "t"}})),
        ("ollama", R(400, {"error": {"message": "m"}})),
        ("openai", R(500, {"error": {"message": "", "type": "t"}})),
    ]:
        try:
            rc.err_ok(r, fam)
        except rc.Fail:
            continue
        raise AssertionError(f"accepted {fam} {r.text}")

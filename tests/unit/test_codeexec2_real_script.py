"""scripts/research/codeexec2_real.py end to end against the in-process gateway with a scripted
model and the real srt: the checks themselves must pass on a well-behaved server before they cost
GPU time (skipped when srt is absent)."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import anthropic
import openai
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_gateway import files_store
from yunshu_gateway.routers import anthropic as anthropic_router
from yunshu_gateway.routers import containers as containers_router
from yunshu_gateway.routers import files as files_router
from yunshu_gateway.routers import responses
from yunshu_gateway.sandbox import containers as cont
from yunshu_gateway.sandbox import executor as exmod
from yunshu_gateway.sandbox import repl as repl_mod
from yunshu_gateway.sandbox.executor import CodeExecutor
from yunshu_gateway.sandbox.srt import find_srt

from .server_tools_helpers import ScriptedInner
from .server_tools_responses_helpers import ScriptedResponses, call, text

ROOT = Path(__file__).resolve().parents[2]
_DEV = Path(
    os.environ.get(
        "YUNSHU_TEST_SRT_BIN", "/Volumes/P5Plus/yunshu-build/srt/node_modules/.bin"
    )
)


def load_script():
    spec = importlib.util.spec_from_file_location(
        "codeexec2_real", ROOT / "scripts/research/codeexec2_real.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["codeexec2_real"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_port_outside_range_is_refused_and_suites_are_registered(monkeypatch):
    mod = load_script()
    assert set(mod.SUITES) == {"small", "big"}
    monkeypatch.setattr(sys, "argv", ["x", "/nonexistent", "--port", "9"])
    with pytest.raises(mod.Fail, match="port"):
        mod.main()


def test_attempt_fails_closed():
    mod = load_script()
    n = []

    def bad():
        n.append(1)
        raise mod.Fail("nope")

    with pytest.raises(mod.Fail):
        mod.attempt(bad, tries=3)
    assert len(n) == 3


@pytest.fixture
def stack(tmp_path, monkeypatch):
    if (_DEV / "srt").exists():
        monkeypatch.setenv("PATH", f"{_DEV}:/opt/homebrew/bin:" + os.environ["PATH"])
    info = find_srt()
    if info is None or info.problem:
        pytest.skip("srt is not available")
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION", "1")
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION_ROOT", str(tmp_path / "sbx"))
    monkeypatch.setenv("YUNSHU_FILES_DIR", str(tmp_path / "files"))
    files_store.reset_store()
    cont.reset_manager()
    monkeypatch.setattr(exmod, "_executor", CodeExecutor(cont.get_manager()))
    app = FastAPI()
    for r in (
        responses.router,
        anthropic_router.router,
        containers_router.router,
        files_router.router,
    ):
        app.include_router(r, prefix="/v1")
    http = TestClient(app)
    http.__enter__()
    yield http, monkeypatch
    http.__exit__(None, None, None)
    repl_mod.get_pool().drop_all()
    cont.reset_manager()


def ci(code):
    return call("code_interpreter", {"code": code})


def test_small_suite_passes_against_a_scripted_server(stack):
    mod = load_script()
    http, mp = stack
    mp.setenv("YUNSHU_CODE_EXECUTION_TIMEOUT", "30")
    oai_rounds = [
        [ci("print(123*456)")],
        [text("done")],
        [ci("print('streamed-ok')")],
        [text("done")],
        [ci("keep_me = 41")],
        [text("ok")],
        [ci("print(keep_me + 1)")],
        [text("42")],
        [ci("open('report.csv','w').write('a,b\\n1,2\\n')")],
        [text("[report.csv](sandbox:/mnt/data/report.csv)")],
        [call("shell", {"commands": ["echo shell-ok && python3 -c 'print(6*7)'"]})],
        [text("done")],
    ]
    mp.setattr(responses, "create_response", ScriptedResponses(oai_rounds))

    def cc(code):
        block = {
            "type": "tool_use",
            "id": "t",
            "name": "code_execution",
            "input": {"code": code},
        }
        return [block], "tool_use"

    ant_rounds = [
        cc("kept = 6"),
        ([{"type": "text", "text": "ok"}], "end_turn"),
        cc("print(kept * 7)"),
        ([{"type": "text", "text": "ok"}], "end_turn"),
        cc("result = await get_weather(city='Paris')\nprint(result)"),
        ([{"type": "text", "text": "Sunny"}], "end_turn"),
    ]
    mp.setattr(anthropic_router, "create_message", ScriptedInner(ant_rounds))
    oai = openai.OpenAI(
        base_url="http://testserver/v1", api_key="k", http_client=http, max_retries=0
    )
    ant = anthropic.Anthropic(
        base_url="http://testserver", api_key="k", http_client=http, max_retries=0
    )
    results: dict = {}
    mod.run_suites("http://testserver", results, False, oai, ant)
    assert all(v["ok"] for v in results.values()), results
    assert set(results) == {
        "containers_api",
        "ci_plain",
        "ci_stream",
        "ci_state_across_responses",
        "ci_file_output",
        "shell_hosted",
        "anthropic_repl_state",
        "anthropic_ptc",
    }


def test_a_failing_step_is_recorded_and_the_run_still_exits_nonzero(monkeypatch):
    mod = load_script()
    results: dict = {}

    class Boom:
        class containers:  # noqa: N801
            @staticmethod
            def create(**kw):
                raise RuntimeError("boom")

    mod.openai_suite("u", results, False, Boom())
    assert results["containers_api"]["ok"] is False
    assert "ci_plain" in results  # later steps ran after the failure


def test_big_suite_passes_against_a_scripted_server(stack):
    mod = load_script()
    http, mp = stack
    mp.setenv("YUNSHU_CODE_EXECUTION_TIMEOUT", "30")
    sha_code = (
        "import hashlib, csv\n"
        "rows = [(w, hashlib.sha256(w.encode()).hexdigest()) for w in ['alpha','beta','gamma','delta']]\n"
        "with open('report.csv','w',newline='') as f:\n"
        "    w = csv.writer(f); w.writerow(['name','sha256']); w.writerows(rows)\n"
        "print(rows[0])"
    )
    oai_rounds = [
        [ci(sha_code)],
        [text("The first row is alpha,...")],
        [call("shell", {"commands": ["echo shell-ok && python3 -c 'print(6*7)'"]})],
        [text("done")],
    ]
    mp.setattr(responses, "create_response", ScriptedResponses(oai_rounds))
    ptc_code = (
        "import asyncio\n"
        "rs = await asyncio.gather(*[get_stock(product=p) for p in ['apple','pear','plum']])\n"
        "print(sum(int(r) for r in rs))"
    )
    block = {
        "type": "tool_use",
        "id": "t",
        "name": "code_execution",
        "input": {"code": ptc_code},
    }
    ant_rounds = [
        ([block], "tool_use"),
        ([{"type": "text", "text": "23"}], "end_turn"),
    ]
    mp.setattr(anthropic_router, "create_message", ScriptedInner(ant_rounds))
    oai = openai.OpenAI(
        base_url="http://testserver/v1", api_key="k", http_client=http, max_retries=0
    )
    ant = anthropic.Anthropic(
        base_url="http://testserver", api_key="k", http_client=http, max_retries=0
    )
    results: dict = {}
    mod.run_suites("http://testserver", results, True, oai, ant)
    assert all(v["ok"] for v in results.values()), results
    assert set(results) == {
        "containers_api",
        "ci_sha256_csv",
        "shell_hosted",
        "anthropic_ptc_parallel",
    }

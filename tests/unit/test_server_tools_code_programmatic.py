"""Anthropic code execution tool versions and programmatic tool calling (code calling the request's
client tools), against the real srt (skipped when it is absent). Blocks are validated with the
anthropic SDK's pydantic types."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from anthropic.types.beta import BetaMessage
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_gateway import files_store
from yunshu_gateway.routers import anthropic
from yunshu_gateway.routers import files as files_router
from yunshu_gateway.sandbox import containers as cont
from yunshu_gateway.sandbox import executor as exmod
from yunshu_gateway.sandbox import repl as repl_mod
from yunshu_gateway.sandbox.executor import CodeExecutor
from yunshu_gateway.sandbox.srt import find_srt
from yunshu_gateway.server_tools.anthropic_loop import (
    drop_programmatic_calls,
    normalize_history,
)

from .server_tools_helpers import ScriptedInner

V = {
    "0522": {"type": "code_execution_20250522", "name": "code_execution"},
    "0825": {"type": "code_execution_20250825", "name": "code_execution"},
    "0120": {"type": "code_execution_20260120", "name": "code_execution"},
    "0521": {"type": "code_execution_20260521", "name": "code_execution"},
}
_DEV = Path(
    os.environ.get(
        "YUNSHU_TEST_SRT_BIN", "/Volumes/P5Plus/yunshu-build/srt/node_modules/.bin"
    )
)
WEATHER = {
    "name": "get_weather",
    "description": "Weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
        "required": ["city"],
    },
}


def prog(caller="code_execution_20260120", **kw):
    return {**WEATHER, "allowed_callers": [caller], **kw}


def call(name, inp, id_="toolu_m1"):
    return {"type": "tool_use", "id": id_, "name": name, "input": inp}


def text(t):
    return {"type": "text", "text": t}


def body(tool=None, tools_extra=(), **kw):
    return {
        "model": "m",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "go"}],
        "tools": [tool or V["0120"], *tools_extra],
        **kw,
    }


@pytest.fixture
def env(tmp_path, monkeypatch):
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
    clients: list[TestClient] = []

    def make(rounds):
        inner = ScriptedInner(rounds)
        monkeypatch.setattr(anthropic, "create_message", inner)
        app = FastAPI()
        app.include_router(anthropic.router, prefix="/v1")
        app.include_router(files_router.router, prefix="/v1")
        c = TestClient(app)
        c.__enter__()  # one event loop for the whole test: the Python process belongs to it
        clients.append(c)
        return c, inner

    yield make
    for c in clients:
        c.__exit__(None, None, None)
    repl_mod.get_pool().drop_all()
    cont.reset_manager()


def names(inner, i=0):
    return [t.name for t in inner.requests[i].tools]


# ── versions ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "ver,expect",
    [
        ("0522", ["code_execution"]),
        ("0825", ["bash_code_execution", "text_editor_code_execution"]),
        (
            "0120",
            ["bash_code_execution", "text_editor_code_execution", "code_execution"],
        ),
        (
            "0521",
            ["bash_code_execution", "text_editor_code_execution", "code_execution"],
        ),
    ],
)
def test_each_version_offers_its_sub_tools(env, ver, expect):
    c, inner = env([([text("hi")], "end_turn")])
    r = c.post("/v1/messages", json=body(V[ver]))
    assert r.status_code == 200, r.text
    assert names(inner) == expect


def test_20250825_gets_python_only_when_code_may_call_tools(env):
    c, inner = env([([text("hi")], "end_turn")])
    c.post(
        "/v1/messages",
        json=body(V["0825"], [prog("code_execution_20250825")]),
    )
    assert names(inner) == [
        "bash_code_execution",
        "text_editor_code_execution",
        "code_execution",
    ]


def test_python_block_shapes_and_state_in_20260120(env):
    c, inner = env(
        [
            ([call("code_execution", {"code": "x = 6"}, "t1")], "tool_use"),
            ([call("code_execution", {"code": "x * 7"}, "t2")], "tool_use"),
            ([text("42")], "end_turn"),
        ]
    )
    m = c.post("/v1/messages", json=body(V["0120"])).json()
    BetaMessage.model_validate(m)
    assert [b["type"] for b in m["content"]] == [
        "server_tool_use",
        "code_execution_tool_result",
        "server_tool_use",
        "code_execution_tool_result",
        "text",
    ]
    assert m["content"][0]["name"] == "code_execution"
    assert m["content"][0]["input"] == {"code": "x = 6"}
    res = m["content"][3]["content"]
    assert res["type"] == "code_execution_result" and res["stdout"] == "42\n"
    assert res["return_code"] == 0 and res["content"] == []


def test_20250522_and_20250825_python_start_fresh_each_call(env):
    for tool, extra in (
        (V["0522"], ()),
        (V["0825"], [prog("code_execution_20250825")]),
    ):
        c, _ = env(
            [
                ([call("code_execution", {"code": "x = 6"}, "t1")], "tool_use"),
                (
                    [call("code_execution", {"code": "print('x' in globals())"}, "t2")],
                    "tool_use",
                ),
                ([text("ok")], "end_turn"),
            ]
        )
        m = c.post("/v1/messages", json=body(tool, extra)).json()
        results = [b for b in m["content"] if b["type"] == "code_execution_tool_result"]
        assert results[1]["content"]["stdout"] == "False\n", tool["type"]


def test_python_files_become_file_ids_and_errors_use_python_error_codes(
    env, monkeypatch
):
    c, _ = env(
        [
            (
                [
                    call(
                        "code_execution",
                        {"code": "open('out.csv','w').write('a,b\\n')"},
                        "t1",
                    )
                ],
                "tool_use",
            ),
            ([call("code_execution", {"code": "while True: pass"}, "t2")], "tool_use"),
            ([text("done")], "end_turn"),
        ]
    )
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION_TIMEOUT", "2")
    m = c.post("/v1/messages", json=body(V["0120"])).json()
    BetaMessage.model_validate(m)
    ok, bad = [b for b in m["content"] if b["type"] == "code_execution_tool_result"]
    fid = ok["content"]["content"][0]["file_id"]
    assert ok["content"]["content"][0]["type"] == "code_execution_output"
    assert (
        c.get(
            f"/v1/files/{fid}/content", headers={"anthropic-version": "2023-06-01"}
        ).content
        == b"a,b\n"
    )
    assert bad["content"] == {
        "type": "code_execution_tool_result_error",
        "error_code": "execution_time_exceeded",
    }


# ── programmatic tool calling ─────────────────────────────────────────────────
CODE = "r = await get_weather(city='SF')\nprint('weather:', r)"


def first_turn(c, tool=None, extra=None, code=CODE, **kw):
    return c.post(
        "/v1/messages",
        json=body(tool, [extra or prog()], **kw),
    ).json()


def test_code_calls_a_client_tool_and_the_turn_pauses(env):
    c, inner = env(
        [
            (
                [text("Let me check."), call("code_execution", {"code": CODE}, "t1")],
                "tool_use",
            )
        ]
    )
    m = first_turn(c)
    BetaMessage.model_validate(m)
    assert m["stop_reason"] == "tool_use"
    kinds = [b["type"] for b in m["content"]]
    assert kinds == ["text", "server_tool_use", "tool_use"]
    stu, tu = m["content"][1], m["content"][2]
    assert tu["name"] == "get_weather" and tu["input"] == {"city": "SF"}
    assert tu["caller"] == {"type": "code_execution_20260120", "tool_id": stu["id"]}
    assert tu["id"].startswith("toolu_") and m["container"]["id"]
    # programmatic-only: the model is not offered the tool as a direct call
    assert "get_weather" not in names(inner)
    # and the python tool tells it about the function
    desc = [t for t in inner.requests[0].tools if t.name == "code_execution"][
        0
    ].description
    assert "await get_weather(city: str, unit: str = None)" in desc


def resume(c, m, result, tid=None, **extra):
    tid = tid or [b for b in m["content"] if b["type"] == "tool_use"][0]["id"]
    return c.post(
        "/v1/messages",
        json=body(
            None,
            [prog()],
            container=m["container"]["id"],
            messages=[
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": m["content"]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tid,
                            "content": result,
                            **extra,
                        }
                    ],
                },
            ],
        ),
    )


def test_resume_finishes_the_cell_and_the_model_sees_one_result(env):
    c, inner = env(
        [
            ([call("code_execution", {"code": CODE}, "t1")], "tool_use"),
            ([text("It is sunny.")], "end_turn"),
        ]
    )
    m1 = first_turn(c)
    r = resume(c, m1, "sunny")
    assert r.status_code == 200, r.text
    m2 = r.json()
    BetaMessage.model_validate(m2)
    assert [b["type"] for b in m2["content"]] == ["code_execution_tool_result", "text"]
    srv = [b for b in m1["content"] if b["type"] == "server_tool_use"][0]
    assert m2["content"][0]["tool_use_id"] == srv["id"]
    assert m2["content"][0]["content"]["stdout"] == "weather: sunny\n"
    assert m2["stop_reason"] == "end_turn"
    # what the model saw: the code call and its final result; never the inner tool call
    seen = inner.requests[-1].messages
    assert [m.role for m in seen] == ["user", "assistant", "user"]
    assert [b["type"] for b in seen[1].content] == ["tool_use"]
    assert seen[1].content[0]["name"] == "code_execution"
    tr = seen[2].content[-1]
    assert tr["type"] == "tool_result" and "weather: sunny" in tr["content"]
    assert not [
        b
        for m in seen
        if isinstance(m.content, list)
        for b in m.content
        if b.get("name") == "get_weather"
    ]


def test_two_sequential_tool_calls_pause_twice(env):
    code = "a = await get_weather(city='SF')\nb = await get_weather(city='NY')\nprint(a, b)"
    c, inner = env(
        [
            ([call("code_execution", {"code": code}, "t1")], "tool_use"),
            ([text("both")], "end_turn"),
        ]
    )
    m1 = first_turn(c)
    r2 = resume(c, m1, "sunny")
    m2 = r2.json()
    assert m2["stop_reason"] == "tool_use"
    assert [b["type"] for b in m2["content"]] == ["tool_use"]
    assert m2["content"][0]["input"] == {"city": "NY"}
    BetaMessage.model_validate(m2)
    # the client's history now has both pauses
    srv = [b for b in m1["content"] if b["type"] == "server_tool_use"][0]
    assert m2["content"][0]["caller"]["tool_id"] == srv["id"]
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": m1["content"]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": m1["content"][-1]["id"],
                    "content": "sunny",
                }
            ],
        },
        {"role": "assistant", "content": m2["content"]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": m2["content"][0]["id"],
                    "content": "rain",
                }
            ],
        },
    ]
    r3 = c.post(
        "/v1/messages",
        json=body(None, [prog()], container=m1["container"]["id"], messages=msgs),
    )
    m3 = r3.json()
    assert m3["content"][0]["content"]["stdout"] == "sunny rain\n"
    seen = inner.requests[-1].messages
    assert [m.role for m in seen] == ["user", "assistant", "user"]


def test_parallel_calls_need_every_result(env):
    code = (
        "import asyncio\nr = await asyncio.gather(get_weather(city='A'), get_weather(city='B'))\n"
        "print(r)"
    )
    c, _ = env(
        [
            ([call("code_execution", {"code": code}, "t1")], "tool_use"),
            ([text("ok")], "end_turn"),
        ]
    )
    m1 = first_turn(c)
    tus = [b for b in m1["content"] if b["type"] == "tool_use"]
    assert sorted(t["input"]["city"] for t in tus) == ["A", "B"]
    bad = resume(c, m1, "x", tid=tus[0]["id"])
    assert bad.status_code == 400 and tus[1]["id"] in bad.json()["error"]["message"]
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": m1["content"]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": t["id"],
                    "content": f"w{t['input']['city']}",
                }
                for t in reversed(tus)
            ],
        },
    ]
    ok = c.post(
        "/v1/messages",
        json=body(None, [prog()], container=m1["container"]["id"], messages=msgs),
    ).json()
    assert ok["content"][0]["content"]["stdout"] == "['wA', 'wB']\n"


def test_tool_error_result_raises_in_the_code(env):
    code = "try:\n    await get_weather(city='x')\nexcept RuntimeError as e:\n    print('failed:', e)"
    c, _ = env(
        [
            ([call("code_execution", {"code": code}, "t1")], "tool_use"),
            ([text("ok")], "end_turn"),
        ]
    )
    m1 = first_turn(c)
    m2 = resume(c, m1, "no such city", is_error=True).json()
    assert m2["content"][0]["content"]["stdout"] == "failed: no such city\n"


def test_resume_after_the_cell_is_gone_is_a_clear_400(env):
    c, _ = env([([call("code_execution", {"code": CODE}, "t1")], "tool_use")])
    m1 = first_turn(c)
    repl_mod.get_pool().drop_all()
    r = resume(c, m1, "sunny")
    assert r.status_code == 400
    assert "no longer running" in r.json()["error"]["message"]


def test_streaming_pause_and_resume_events(env):
    c, _ = env(
        [
            ([call("code_execution", {"code": CODE}, "t1")], "tool_use"),
            ([text("sunny")], "end_turn"),
        ]
    )
    with c.stream("POST", "/v1/messages", json=body(None, [prog()], stream=True)) as r:
        raw = r.read().decode()
    ev = [
        (b.split("\n")[0][7:], json.loads(b.split("\n")[1][6:]))
        for b in raw.strip().split("\n\n")
    ]
    starts = [e["content_block"] for n, e in ev if n == "content_block_start"]
    assert [b["type"] for b in starts] == ["server_tool_use", "tool_use"]
    assert starts[1]["caller"]["type"] == "code_execution_20260120"
    idx = starts[1]
    deltas = [
        e["delta"]["partial_json"]
        for n, e in ev
        if n == "content_block_delta" and e["index"] == 1
    ]
    assert json.loads("".join(deltas)) == {"city": "SF"}
    md = [e for n, e in ev if n == "message_delta"][0]
    assert md["delta"]["stop_reason"] == "tool_use" and md["delta"]["container"]["id"]
    assert idx["name"] == "get_weather"
    from anthropic.types.beta import BetaRawMessageStreamEvent
    from pydantic import TypeAdapter

    ta = TypeAdapter(BetaRawMessageStreamEvent)
    for _n, e in ev:
        ta.validate_python(e)


def test_direct_and_programmatic_tool_is_also_offered_directly(env):
    c, inner = env([([text("hi")], "end_turn")])
    t = {**WEATHER, "allowed_callers": ["direct", "code_execution_20260120"]}
    c.post("/v1/messages", json=body(None, [t]))
    assert "get_weather" in names(inner)


def test_allowed_callers_without_the_code_tool_is_refused(env):
    c, _ = env([])
    r = c.post(
        "/v1/messages",
        json={
            "model": "m",
            "max_tokens": 9,
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}, prog()],
        },
    )
    assert r.status_code == 400
    assert "allowed_callers" in r.json()["error"]["message"]


# ── history (CPU only) ────────────────────────────────────────────────────────
def _msgs():
    return [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_1",
                    "name": "code_execution",
                    "input": {"code": "x"},
                },
                {
                    "type": "tool_use",
                    "id": "toolu_a",
                    "name": "get_weather",
                    "input": {},
                    "caller": {
                        "type": "code_execution_20260120",
                        "tool_id": "srvtoolu_1",
                    },
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_a", "content": "sunny"}
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "code_execution_tool_result",
                    "tool_use_id": "srvtoolu_1",
                    "content": {
                        "type": "code_execution_result",
                        "stdout": "ok\n",
                        "stderr": "",
                        "return_code": 0,
                        "content": [],
                    },
                },
                {"type": "text", "text": "It is sunny."},
            ],
        },
    ]


def test_completed_programmatic_turn_replays_as_one_code_call():
    h = normalize_history(_msgs())
    assert [m["role"] for m in h] == ["user", "assistant", "user", "assistant"]
    assert [b["type"] for b in h[1]["content"]] == ["text", "tool_use"]
    assert h[1]["content"][1]["name"] == "code_execution"
    assert h[2]["content"][0]["type"] == "tool_result"
    assert "stdout:\nok" in h[2]["content"][0]["content"]
    assert h[3]["content"][0]["text"] == "It is sunny."


def test_drop_programmatic_calls_is_identity_without_them():
    plain = [{"role": "user", "content": "x"}]
    assert drop_programmatic_calls(plain) is plain

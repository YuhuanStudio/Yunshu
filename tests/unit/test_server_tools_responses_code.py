"""Responses API code_interpreter and hosted shell, against the real srt (skipped when it is absent).

Output items and streaming events are validated with the openai SDK's pydantic types."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai.types.responses import Response
from openai.types.responses.response_stream_event import ResponseStreamEvent
from pydantic import TypeAdapter

from yunshu_gateway import files_store
from yunshu_gateway.routers import containers as containers_router
from yunshu_gateway.routers import responses
from yunshu_gateway.sandbox import containers as cont
from yunshu_gateway.sandbox import executor as exmod
from yunshu_gateway.sandbox import repl as repl_mod
from yunshu_gateway.sandbox.executor import CodeExecutor
from yunshu_gateway.sandbox.srt import find_srt
from yunshu_gateway.server_tools.responses_loop import (
    container_file_annotations,
    input_item_to_messages,
)

from .server_tools_responses_helpers import ScriptedResponses, call, parse_sse, text

_DEV = Path(
    os.environ.get(
        "YUNSHU_TEST_SRT_BIN", "/Volumes/P5Plus/yunshu-build/srt/node_modules/.bin"
    )
)
EVENT = TypeAdapter(ResponseStreamEvent)
CI = {"type": "code_interpreter", "container": {"type": "auto"}}
SH = {"type": "shell", "environment": {"type": "container_auto"}}


@pytest.fixture
def srt_ok(monkeypatch):
    if (_DEV / "srt").exists():
        monkeypatch.setenv("PATH", f"{_DEV}:/opt/homebrew/bin:" + os.environ["PATH"])
    info = find_srt()
    if info is None or info.problem:
        pytest.skip("srt is not available")


@pytest.fixture
def make(tmp_path, monkeypatch, srt_ok):
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION", "1")
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION_ROOT", str(tmp_path / "sbx"))
    monkeypatch.setenv("YUNSHU_FILES_DIR", str(tmp_path / "files"))
    files_store.reset_store()
    cont.reset_manager()
    monkeypatch.setattr(exmod, "_executor", CodeExecutor(cont.get_manager()))

    def _make(rounds):
        inner = ScriptedResponses(rounds)
        monkeypatch.setattr(responses, "create_response", inner)
        app = FastAPI()
        app.include_router(responses.router, prefix="/v1")
        app.include_router(containers_router.router, prefix="/v1")
        client = TestClient(app)
        client.__enter__()  # one event loop for every request (the REPL belongs to it)
        clients.append(client)
        return client, inner

    clients: list[TestClient] = []
    yield _make
    for cl in clients:
        cl.__exit__(None, None, None)
    repl_mod.get_pool().drop_all()
    cont.reset_manager()


def body(tools, **kw):
    return {"model": "m", "input": "compute", "tools": tools, "store": True, **kw}


def types(r):
    return [i["type"] for i in r["output"]]


def test_code_interpreter_runs_python_with_state_and_validates(make):
    c, inner = make(
        [
            [call("code_interpreter", {"code": "x = 6\nprint('a')"})],
            [call("code_interpreter", {"code": "x * 7"})],
            [text("The answer is 42.")],
        ]
    )
    r = c.post(
        "/v1/responses",
        json=body([CI], include=["code_interpreter_call.outputs"]),
    ).json()
    Response.model_validate(r)
    assert types(r) == ["code_interpreter_call", "code_interpreter_call", "message"]
    a, b, _ = r["output"]
    assert a["status"] == "completed" and a["code"] == "x = 6\nprint('a')"
    assert a["container_id"].startswith("cntr_") and a["id"].startswith("ci_")
    assert a["outputs"] == [{"type": "logs", "logs": "a\n"}]
    assert b["container_id"] == a["container_id"]  # one container for the response
    assert b["outputs"] == [{"type": "logs", "logs": "42\n"}]  # state persisted
    # the model saw the results as function outputs
    last = inner.requests[-1].input
    outs = [
        getattr(i, "output", None) for i in last if i.type == "function_call_output"
    ]
    assert any("42" in (o or "") for o in outs)
    # the container is a real /v1/containers container
    got = c.get(f"/v1/containers/{a['container_id']}")
    assert got.status_code == 200


def test_outputs_are_null_unless_included(make):
    c, _ = make([[call("code_interpreter", {"code": "print(1)"})], [text("ok")]])
    r = c.post("/v1/responses", json=body([CI])).json()
    Response.model_validate(r)
    assert r["output"][0]["outputs"] is None and r["output"][0]["status"] == "completed"


def test_python_error_is_a_completed_call_the_model_can_read(make):
    c, inner = make(
        [
            [call("code_interpreter", {"code": "1/0"})],
            [text("it failed")],
        ]
    )
    r = c.post(
        "/v1/responses", json=body([CI], include=["code_interpreter_call.outputs"])
    ).json()
    item = r["output"][0]
    assert "ZeroDivisionError" in item["outputs"][0]["logs"]
    out = [i for i in inner.requests[-1].input if i.type == "function_call_output"][0]
    assert "ZeroDivisionError" in out.output


def test_timeout_fails_the_call_and_next_call_gets_a_clean_process(make, monkeypatch):
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION_TIMEOUT", "2")
    c, _ = make(
        [
            [call("code_interpreter", {"code": "v = 1"})],
            [call("code_interpreter", {"code": "while True: pass"})],
            [call("code_interpreter", {"code": "print('v' in globals())"})],
            [text("done")],
        ]
    )
    r = c.post(
        "/v1/responses", json=body([CI], include=["code_interpreter_call.outputs"])
    ).json()
    s = [i["status"] for i in r["output"][:3]]
    assert s == ["completed", "failed", "completed"]
    assert "False" in r["output"][2]["outputs"][0]["logs"]


def test_streaming_events_validate_and_are_ordered(make):
    c, _ = make(
        [
            [call("code_interpreter", {"code": "print('hi')"})],
            [text("done")],
        ]
    )
    with c.stream("POST", "/v1/responses", json=body([CI], stream=True)) as resp:
        raw = resp.read().decode()
    evs = parse_sse(raw)
    for _n, e in evs:
        EVENT.validate_python(e)
    names = [n for n, _ in evs]
    seq = [n for n in names if "code_interpreter" in n]
    assert seq == [
        "response.code_interpreter_call.in_progress",
        "response.code_interpreter_call_code.delta",
        "response.code_interpreter_call_code.done",
        "response.code_interpreter_call.interpreting",
        "response.code_interpreter_call.completed",
    ]
    nums = [e["sequence_number"] for _, e in evs]
    assert nums == sorted(set(nums))
    added = [e for n, e in evs if n == "response.output_item.added"]
    assert added[0]["item"]["type"] == "code_interpreter_call"
    assert [e for n, e in evs if n == "response.completed"]


def test_created_file_is_cited_and_downloadable(make):
    c, _ = make(
        [
            [
                call(
                    "code_interpreter",
                    {"code": "open('/mnt/data/report.csv','w').write('a,b\\n1,2\\n')"},
                )
            ],
            [text("Here you go: [report.csv](sandbox:/mnt/data/report.csv) done.")],
        ]
    )
    r = c.post("/v1/responses", json=body([CI])).json()
    Response.model_validate(r)
    msg = r["output"][-1]
    part = msg["content"][0]
    cites = [a for a in part["annotations"] if a["type"] == "container_file_citation"]
    assert cites and cites[0]["filename"] == "report.csv"
    cid = r["output"][0]["container_id"]
    assert cites[0]["container_id"] == cid
    spans = [part["text"][a["start_index"] : a["end_index"]] for a in cites]
    assert spans == ["report.csv", "sandbox:/mnt/data/report.csv"]
    dl = c.get(f"/v1/containers/{cid}/files/{cites[0]['file_id']}/content")
    assert dl.status_code == 200 and dl.content == b"a,b\n1,2\n"


def test_explicit_container_id_and_unknown_id(make):
    c, _ = make([[call("code_interpreter", {"code": "z = 5"})], [text("ok")]])
    cid = c.post("/v1/containers", json={"name": "mine"}).json()["id"]
    r = c.post(
        "/v1/responses", json=body([{"type": "code_interpreter", "container": cid}])
    ).json()
    assert r["output"][0]["container_id"] == cid
    bad = c.post(
        "/v1/responses",
        json=body([{"type": "code_interpreter", "container": "cntr_" + "0" * 24}]),
    )
    assert (
        bad.status_code == 400 and bad.json()["error"]["code"] == "container_not_found"
    )


def test_auto_container_with_file_ids_stages_files(make):
    c, _ = make(
        [
            [call("code_interpreter", {"code": "print(open('d.txt').read())"})],
            [text("ok")],
        ]
    )
    meta = files_store.get_store().put(b"payload", "d.txt", "user_data")
    r = c.post(
        "/v1/responses",
        json=body(
            [
                {
                    "type": "code_interpreter",
                    "container": {"type": "auto", "file_ids": [meta["id"]]},
                }
            ],
            include=["code_interpreter_call.outputs"],
        ),
    ).json()
    assert r["output"][0]["outputs"][0]["logs"] == "payload\n"


def test_previous_response_reuses_the_container(make):
    c, _ = make(
        [
            [call("code_interpreter", {"code": "keep = 99"})],
            [text("a")],
            [call("code_interpreter", {"code": "print(keep)"})],
            [text("b")],
        ]
    )
    r1 = c.post("/v1/responses", json=body([CI])).json()
    r2 = c.post(
        "/v1/responses",
        json=body(
            [CI],
            previous_response_id=r1["id"],
            include=["code_interpreter_call.outputs"],
        ),
    ).json()
    assert r2["output"][0]["container_id"] == r1["output"][0]["container_id"]
    assert r2["output"][0]["outputs"][0]["logs"] == "99\n"


def test_disabled_is_400_and_missing_srt_is_503(make, monkeypatch):
    c, _ = make([])
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION", "0")
    r = c.post("/v1/responses", json=body([CI]))
    assert (
        r.status_code == 400 and "YUNSHU_CODE_EXECUTION" in r.json()["error"]["message"]
    )
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION", "1")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(Path, "home", lambda: Path("/nonexistent"))
    r = c.post("/v1/responses", json=body([CI]))
    assert r.status_code == 503


# ── hosted shell ──────────────────────────────────────────────────────────────
def test_shell_runs_commands_and_emits_call_and_output_items(make):
    c, inner = make(
        [
            [
                call(
                    "shell",
                    {
                        "commands": ["echo hi", "echo err >&2; exit 3", "sleep 5"],
                        "timeout_ms": 1500,
                    },
                )
            ],
            [text("done")],
        ]
    )
    r = c.post("/v1/responses", json=body([SH])).json()
    Response.model_validate(r)
    assert types(r) == ["shell_call", "shell_call_output", "message"]
    call_, out, _ = r["output"]
    assert call_["action"]["commands"] == ["echo hi", "echo err >&2; exit 3", "sleep 5"]
    assert call_["environment"]["type"] == "container_reference"
    assert call_["environment"]["container_id"].startswith("cntr_")
    assert out["call_id"] == call_["call_id"] and out["status"] == "completed"
    o = out["output"]
    assert o[0] == {
        "stdout": "hi\n",
        "stderr": "",
        "outcome": {"type": "exit", "exit_code": 0},
    }
    assert o[1]["stderr"] == "err\n" and o[1]["outcome"] == {
        "type": "exit",
        "exit_code": 3,
    }
    assert o[2]["outcome"] == {"type": "timeout"}
    fo = [i for i in inner.requests[-1].input if i.type == "function_call_output"][0]
    assert "exit code: 3" in fo.output and "[timed out]" in fo.output


def test_shell_streaming_events_validate(make):
    c, _ = make([[call("shell", {"commands": ["echo a", "echo b"]})], [text("ok")]])
    with c.stream("POST", "/v1/responses", json=body([SH], stream=True)) as resp:
        raw = resp.read().decode()
    evs = parse_sse(raw)
    for _n, e in evs:
        EVENT.validate_python(e)
    names = [n for n, _ in evs]
    assert names.count("response.shell_call_command.added") == 2
    assert names.count("response.shell_call_command.done") == 2
    assert names.count("response.shell_call_output_content.delta") == 2
    assert names.count("response.shell_call_output_content.done") == 2
    i_call = names.index("response.shell_call_command.added")
    i_out = names.index("response.shell_call_output_content.delta")
    assert i_call < i_out


def test_shell_and_code_interpreter_share_the_container(make):
    c, _ = make(
        [
            [call("shell", {"commands": ["echo shared > s.txt"]})],
            [call("code_interpreter", {"code": "print(open('s.txt').read())"})],
            [text("ok")],
        ]
    )
    r = c.post(
        "/v1/responses",
        json=body([SH, CI], include=["code_interpreter_call.outputs"]),
    ).json()
    assert "output" in r, r
    ci = [i for i in r["output"] if i["type"] == "code_interpreter_call"][0]
    sh = [i for i in r["output"] if i["type"] == "shell_call"][0]
    assert ci["container_id"] == sh["environment"]["container_id"]
    assert ci["outputs"][0]["logs"] == "shared\n\n"


def test_shell_container_reference(make):
    c, _ = make([[call("shell", {"commands": ["pwd"]})], [text("ok")]])
    cid = c.post("/v1/containers", json={"name": "r"}).json()["id"]
    r = c.post(
        "/v1/responses",
        json=body(
            [
                {
                    "type": "shell",
                    "environment": {"type": "container_reference", "container_id": cid},
                }
            ]
        ),
    ).json()
    assert r["output"][0]["environment"]["container_id"] == cid
    assert cid in r["output"][1]["output"][0]["stdout"]


def test_two_containers_in_one_request_are_refused(make):
    c, _ = make([])
    a = c.post("/v1/containers", json={"name": "a"}).json()["id"]
    b = c.post("/v1/containers", json={"name": "b"}).json()["id"]
    r = c.post(
        "/v1/responses",
        json=body(
            [
                {"type": "code_interpreter", "container": a},
                {
                    "type": "shell",
                    "environment": {"type": "container_reference", "container_id": b},
                },
            ]
        ),
    )
    assert r.status_code == 400


def test_local_shell_environment_is_not_served_by_the_server(make):
    # environment local / absent means the CLIENT runs commands: not a server tool
    from yunshu_gateway.server_tools.responses_loop import has_server_tools_responses

    class R:
        tools = [{"type": "shell", "environment": {"type": "local"}}]

    assert not has_server_tools_responses(R())
    R.tools = [{"type": "shell"}]
    assert not has_server_tools_responses(R())


# ── replay and annotations (CPU only) ─────────────────────────────────────────
def test_history_replay_of_code_items():
    ci = {
        "id": "ci_1",
        "type": "code_interpreter_call",
        "code": "print(1)",
        "container_id": "cntr_x",
        "outputs": [{"type": "logs", "logs": "1\n"}],
        "status": "completed",
    }
    m = input_item_to_messages(ci)
    assert m[0]["tool_calls"][0]["function"]["name"] == "code_interpreter"
    assert m[1]["content"] == "1\n"
    call_ = {"type": "shell_call", "call_id": "call_1", "action": {"commands": ["ls"]}}
    out = {
        "type": "shell_call_output",
        "call_id": "call_1",
        "output": [
            {"stdout": "a\n", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}
        ],
    }
    mc, mo = input_item_to_messages(call_)[0], input_item_to_messages(out)[0]
    assert mc["tool_calls"][0]["id"] == mo["tool_call_id"] == "call_1"
    assert "exit code: 0" in mo["content"] and "a" in mo["content"]


def test_container_file_annotations_spans():
    created = {
        "a/b.csv": {
            "id": "cfile_1",
            "container_id": "cntr_1",
            "path": "/mnt/data/a/b.csv",
        },
        "b.csv": {"id": "cfile_2", "container_id": "cntr_1", "path": "/mnt/data/b.csv"},
    }
    t = "See sandbox:/mnt/data/a/b.csv and b.csv, not xb.csv."
    anns = container_file_annotations(t, created)
    spans = [(a["file_id"], t[a["start_index"] : a["end_index"]]) for a in anns]
    assert spans == [
        ("cfile_1", "sandbox:/mnt/data/a/b.csv"),
        ("cfile_2", "b.csv"),
    ]


def test_client_shell_is_a_function_and_its_call_item_validates():
    import json

    from openai.types.responses import ResponseFunctionShellToolCall

    from yunshu_gateway.responses_client_tools import call_item, declarations
    from yunshu_gateway.server_tools.responses_loop import function_tools

    q = responses.ResponsesRequest(
        model="m",
        input="x",
        tools=[{"type": "shell", "environment": {"type": "local"}}],
    )
    tools = function_tools(q.tools)
    assert [t.name for t in tools] == ["shell"]
    assert tools[0].parameters["properties"]["commands"]["type"] == "array"
    item = call_item(
        {
            "id": "fc_1",
            "call_id": "call_1",
            "name": "shell",
            "arguments": json.dumps({"commands": ["ls"], "timeout_ms": 5}),
            "status": "completed",
        },
        declarations(q.tools)["shell"],
    )
    ResponseFunctionShellToolCall.model_validate(item)
    assert item["action"] == {"commands": ["ls"], "timeout_ms": 5}
    assert item["environment"] == {"type": "local"}


def test_hosted_shell_is_not_a_client_function():
    from yunshu_gateway.server_tools.responses_loop import function_tools

    q = responses.ResponsesRequest(model="m", input="x", tools=[SH])
    assert function_tools(q.tools) is None


def test_streaming_file_citation_events_validate(make):
    c, _ = make(
        [
            [call("code_interpreter", {"code": "open('r.txt','w').write('x')"})],
            [text("Saved as sandbox:/mnt/data/r.txt for you.")],
        ]
    )
    with c.stream("POST", "/v1/responses", json=body([CI], stream=True)) as resp:
        raw = resp.read().decode()
    evs = parse_sse(raw)
    for _n, e in evs:
        EVENT.validate_python(e)
    added = [e for n, e in evs if n == "response.output_text.annotation.added"]
    assert added and added[0]["annotation"]["type"] == "container_file_citation"
    done = [e for n, e in evs if n == "response.output_item.done"][-1]["item"]
    assert done["content"][0]["annotations"][0]["filename"] == "r.txt"


def test_tool_choice_for_a_hosted_code_tool_forces_its_function(make):
    c, inner = make([[call("code_interpreter", {"code": "1"})], [text("ok")]])
    c.post(
        "/v1/responses",
        json=body([CI], tool_choice={"type": "code_interpreter"}),
    )
    tc = inner.requests[0].tool_choice
    assert tc == {"type": "function", "name": "code_interpreter"}
    assert inner.requests[1].tool_choice == "auto"  # only the first round is forced


def test_skills_in_an_auto_environment_are_refused(make):
    c, _ = make([])
    r = c.post(
        "/v1/responses",
        json=body(
            [
                {
                    "type": "shell",
                    "environment": {
                        "type": "container_auto",
                        "skills": [{"type": "skill_reference", "skill_id": "s"}],
                    },
                }
            ]
        ),
    )
    assert r.status_code == 400 and r.json()["error"]["code"] == "unsupported_parameter"

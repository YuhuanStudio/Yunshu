"""Anthropic code_execution_* server tool: gating, block shapes (validated with the anthropic 2.x
SDK's pydantic types), streaming, container reuse, history replay, uploads, and a real-srt run."""

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
from yunshu_gateway.sandbox.executor import CodeExecutor, ExecResult
from yunshu_gateway.sandbox.srt import SrtInfo, find_srt
from yunshu_gateway.server_tools.anthropic_loop import normalize_history

from .server_tools_helpers import ScriptedInner

CE = {"type": "code_execution_20250825", "name": "code_execution"}
_DEV = Path(
    os.environ.get(
        "YUNSHU_TEST_SRT_BIN", "/Volumes/P5Plus/yunshu-build/srt/node_modules/.bin"
    )
)


def call(name, inp, id_="toolu_m1"):
    return {"type": "tool_use", "id": id_, "name": name, "input": inp}


def text(t):
    return {"type": "text", "text": t}


def body(**kw):
    return {
        "model": "m",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "compute"}],
        "tools": [CE],
        **kw,
    }


class FakeExec(CodeExecutor):
    """Real containers and editor, scripted bash (no srt)."""

    def __init__(self, manager, results=None):
        super().__init__(manager)
        self.results = list(results or [])
        self.commands: list[str] = []

    def srt(self):
        return SrtInfo("/n", "/c")

    async def bash(self, c, command, timeout=None):
        self.commands.append(command)
        return self.results.pop(0) if self.results else ExecResult(stdout="ok\n")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION", "1")
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION_ROOT", str(tmp_path / "sbx"))
    monkeypatch.setenv("YUNSHU_FILES_DIR", str(tmp_path / "files"))
    files_store.reset_store()
    cont.reset_manager()
    fake = FakeExec(cont.get_manager())
    monkeypatch.setattr(exmod, "_executor", fake)

    def make(rounds):
        inner = ScriptedInner(rounds)
        monkeypatch.setattr(anthropic, "create_message", inner)
        app = FastAPI()
        app.include_router(anthropic.router, prefix="/v1")
        return TestClient(app), inner

    yield make, fake
    cont.reset_manager()


def test_off_is_400_and_missing_srt_is_503(env, monkeypatch):
    make, fake = env
    c, _ = make([])
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION", "0")
    r = c.post("/v1/messages", json=body())
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
    assert "YUNSHU_CODE_EXECUTION" in r.json()["error"]["message"]
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION", "1")
    monkeypatch.setattr(exmod, "find_srt", lambda: None)
    monkeypatch.setattr(exmod, "_executor", CodeExecutor(cont.get_manager()))
    r = c.post("/v1/messages", json=body())
    assert r.status_code == 503 and r.json()["error"]["type"] == "api_error"
    assert "sandbox-runtime" in r.json()["error"]["message"]


def test_bash_block_shapes_validate_with_sdk(env):
    make, fake = env
    fake.results = [
        ExecResult(
            stdout="42\n",
            stderr="warn\n",
            return_code=0,
            files=[{"file_id": "file_x", "filename": "r.csv"}],
        )
    ]
    c, inner = make(
        [
            (
                [
                    text("Running."),
                    call("bash_code_execution", {"command": "python3 -c 'print(42)'"}),
                ],
                "tool_use",
            ),
            ([text("It is 42.")], "end_turn"),
        ]
    )
    r = c.post("/v1/messages", json=body())
    assert r.status_code == 200, r.text
    m = r.json()
    assert [b["type"] for b in m["content"]] == [
        "text",
        "server_tool_use",
        "bash_code_execution_tool_result",
        "text",
    ]
    stu, res = m["content"][1], m["content"][2]
    assert stu["name"] == "bash_code_execution" and stu["id"].startswith("srvtoolu_")
    assert res["tool_use_id"] == stu["id"]
    assert res["content"] == {
        "type": "bash_code_execution_result",
        "stdout": "42\n",
        "stderr": "warn\n",
        "return_code": 0,
        "content": [{"type": "bash_code_execution_output", "file_id": "file_x"}],
    }
    assert cont.valid_id(m["container"]["id"]) and m["container"][
        "expires_at"
    ].endswith("Z")
    sdk = BetaMessage.model_validate(m)
    assert sdk.container.id == m["container"]["id"]
    assert sdk.content[2].content.return_code == 0
    # the model saw the result text and the tools as plain functions
    assert [t.name for t in inner.requests[0].tools] == [
        "bash_code_execution",
        "text_editor_code_execution",
    ]
    tr = inner.requests[1].messages[2].content[0]
    assert tr["type"] == "tool_result" and "stdout:\n42" in tr["content"]
    assert inner.requests[0].container is None


def test_editor_blocks_and_errors(env):
    make, fake = env
    c, _ = make(
        [
            (
                [
                    call(
                        "text_editor_code_execution",
                        {"command": "create", "path": "a.txt", "file_text": "x\ny\n"},
                        "t1",
                    ),
                    call(
                        "text_editor_code_execution",
                        {"command": "view", "path": "a.txt"},
                        "t2",
                    ),
                    call(
                        "text_editor_code_execution",
                        {"command": "view", "path": "missing"},
                        "t3",
                    ),
                ],
                "tool_use",
            ),
            ([text("done")], "end_turn"),
        ]
    )
    m = c.post("/v1/messages", json=body()).json()
    results = [
        b for b in m["content"] if b["type"] == "text_editor_code_execution_tool_result"
    ]
    assert [r["content"]["type"] for r in results] == [
        "text_editor_code_execution_create_result",
        "text_editor_code_execution_view_result",
        "text_editor_code_execution_tool_result_error",
    ]
    assert (
        results[1]["content"]["content"] == "x\ny"
        and results[2]["content"]["error_code"] == "file_not_found"
    )
    BetaMessage.model_validate(m)


def test_error_result_shapes(env):
    make, fake = env
    fake.results = [ExecResult(False, "execution_time_exceeded", text="exceeded 1 s")]
    c, _ = make(
        [
            ([call("bash_code_execution", {"command": "sleep 9"})], "tool_use"),
            ([text("slow")], "end_turn"),
        ]
    )
    m = c.post("/v1/messages", json=body()).json()
    res = m["content"][1]
    assert res["content"] == {
        "type": "bash_code_execution_tool_result_error",
        "error_code": "execution_time_exceeded",
    }
    BetaMessage.model_validate(m)


def test_stream_events_and_container(env):
    make, fake = env
    c, _ = make(
        [
            ([call("bash_code_execution", {"command": "echo hi"})], "tool_use"),
            ([text("hi")], "end_turn"),
        ]
    )
    with c.stream("POST", "/v1/messages", json=body(stream=True)) as r:
        raw = r.read().decode()
    ev = [
        (b.split("\n")[0][7:], json.loads(b.split("\n")[1][6:]))
        for b in raw.strip().split("\n\n")
    ]
    starts = [
        (e["index"], e["content_block"]["type"])
        for n, e in ev
        if n == "content_block_start"
    ]
    assert starts == [
        (0, "server_tool_use"),
        (1, "bash_code_execution_tool_result"),
        (2, "text"),
    ]
    deltas = [e for n, e in ev if n == "content_block_delta" and e["index"] == 0]
    assert json.loads("".join(d["delta"]["partial_json"] for d in deltas)) == {
        "command": "echo hi"
    }
    # the result block arrives whole in its content_block_start
    assert (
        ev[[n for n, _ in ev].index("content_block_start", 3)][1]["content_block"][
            "content"
        ]["stdout"]
        == "ok\n"
    )
    md = [e for n, e in ev if n == "message_delta"][0]
    assert (
        cont.valid_id(md["delta"]["container"]["id"])
        and md["delta"]["stop_reason"] == "end_turn"
    )


def test_sdk_stream_accumulates_container(env):
    # the SDK's own accumulator must accept our events
    from anthropic.lib.streaming._beta_messages import accumulate_event  # type: ignore

    make, fake = env
    c, _ = make(
        [
            ([call("bash_code_execution", {"command": "echo hi"})], "tool_use"),
            ([text("hi")], "end_turn"),
        ]
    )
    with c.stream("POST", "/v1/messages", json=body(stream=True)) as r:
        raw = r.read().decode()
    from anthropic.types.beta import BetaRawMessageStreamEvent
    from pydantic import TypeAdapter

    ta = TypeAdapter(BetaRawMessageStreamEvent)
    snap = None
    bufs: dict = {}
    import httpx

    for blk in raw.strip().split("\n\n"):
        e = ta.validate_python(json.loads(blk.split("\n")[1][6:]))
        snap = accumulate_event(
            event=e,
            current_snapshot=snap,
            json_bufs=bufs,
            request_headers=httpx.Headers({}),
        )
    assert (
        snap.container is not None
        and snap.content[1].type == "bash_code_execution_tool_result"
    )


def test_container_reuse_unknown_and_expired(env, monkeypatch):
    make, fake = env
    c, _ = make(
        [
            ([call("bash_code_execution", {"command": "echo a"})], "tool_use"),
            ([text("a")], "end_turn"),
        ]
    )
    cid = c.post("/v1/messages", json=body()).json()["container"]["id"]
    fake.container_seen = None
    c, _ = make(
        [
            (
                [
                    call(
                        "text_editor_code_execution",
                        {"command": "create", "path": "n", "file_text": "1"},
                    )
                ],
                "tool_use",
            ),
            ([text("b")], "end_turn"),
        ]
    )
    m = c.post("/v1/messages", json=body(container=cid)).json()
    assert m["container"]["id"] == cid
    assert (fake.manager.get(cid).work / "n").read_text() == "1"
    m2 = c.post("/v1/messages", json=body(container={"id": cid})).json()
    assert m2["container"]["id"] == cid
    r = c.post("/v1/messages", json=body(container="container_" + "0" * 24))
    assert r.status_code == 400 and "not found" in r.json()["error"]["message"]
    mgr = fake.manager
    t = mgr.clock()
    monkeypatch.setattr(mgr, "clock", lambda: t + 10_000)
    r = c.post("/v1/messages", json=body(container=cid))
    assert r.status_code == 400 and "expired" in r.json()["error"]["message"]
    assert c.post("/v1/messages", json=body(container="../etc")).status_code == 400


def test_history_replay_of_code_blocks(env):
    msgs = [
        {"role": "user", "content": "run it"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_1",
                    "name": "bash_code_execution",
                    "input": {"command": "echo 1"},
                },
                {
                    "type": "bash_code_execution_tool_result",
                    "tool_use_id": "srvtoolu_1",
                    "content": {
                        "type": "bash_code_execution_result",
                        "stdout": "1\n",
                        "stderr": "",
                        "return_code": 0,
                        "content": [],
                    },
                },
                {"type": "text", "text": "It printed 1."},
            ],
        },
        {"role": "user", "content": "again"},
    ]
    out = normalize_history(msgs)
    assert [m["role"] for m in out] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert out[1]["content"][0] == {
        "type": "tool_use",
        "id": "srvtoolu_1",
        "name": "bash_code_execution",
        "input": {"command": "echo 1"},
    }
    assert out[2]["content"][0]["content"] == "exit code: 0\nstdout:\n1\n"


def test_container_upload_is_staged(env):
    make, fake = env
    meta = files_store.get_store().put(b"a,b\n1,2\n", "data.csv")
    c, inner = make([([text("ok")], "end_turn")])
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "analyse"},
                {"type": "container_upload", "file_id": meta["id"]},
            ],
        }
    ]
    m = c.post("/v1/messages", json=body(messages=msgs)).json()
    cid = m["container"]["id"] if "container" in m else None
    # a container exists even though no command ran, and holds the file
    boxes = [d for d in fake.manager.root.iterdir() if cont.valid_id(d.name)]
    assert (
        len(boxes) == 1
        and (boxes[0] / "work" / "data.csv").read_bytes() == b"a,b\n1,2\n"
    )
    seen = inner.requests[0].messages[0].content
    assert "data.csv" in json.dumps(seen) and "container_upload" not in json.dumps(seen)
    assert cid is None or cont.valid_id(cid)
    r = c.post(
        "/v1/messages",
        json=body(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "container_upload", "file_id": "file_" + "0" * 24}
                    ],
                }
            ]
        ),
    )
    assert r.status_code == 404


def test_forced_code_execution_maps_to_bash(env):
    make, _ = env
    c, inner = make([([text("x")], "end_turn")])
    c.post(
        "/v1/messages",
        json=body(tool_choice={"type": "tool", "name": "code_execution"}),
    )
    assert inner.requests[0].tool_choice == {
        "type": "tool",
        "name": "bash_code_execution",
    }


# ── real srt ────────────────────────────────────────────────────────────────
@pytest.fixture
def real(tmp_path, monkeypatch):
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
    monkeypatch.setattr(exmod, "_executor", CodeExecutor())

    def make(rounds):
        inner = ScriptedInner(rounds)
        monkeypatch.setattr(anthropic, "create_message", inner)
        app = FastAPI()
        app.include_router(anthropic.router, prefix="/v1")
        app.include_router(files_router.router, prefix="/v1")
        return TestClient(app), inner

    yield make
    cont.reset_manager()


def test_real_srt_file_output_roundtrip(real):
    cmd = "printf 'sha\\n' > \"$OUTPUT_DIR/out.csv\" && python3 -c 'import hashlib;print(hashlib.sha256(b\"abc\").hexdigest())'"
    c, _ = real(
        [
            ([call("bash_code_execution", {"command": cmd})], "tool_use"),
            ([text("done")], "end_turn"),
        ]
    )
    m = c.post("/v1/messages", json=body()).json()
    res = m["content"][1]["content"]
    assert (
        res["stdout"].strip()
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    fid = res["content"][0]["file_id"]
    assert files_store.get_store().read(fid) == b"sha\n"
    BetaMessage.model_validate(m)


def test_real_server_script_logic_against_scripted_model(real):
    """scripts/research/codeexec_real.py end to end over the SDK, with a scripted model and real srt."""
    import hashlib
    import importlib.util

    import anthropic

    spec = importlib.util.spec_from_file_location(
        "codeexec_real",
        Path(__file__).resolve().parents[2] / "scripts/research/codeexec_real.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def bash(cmd, i):
        return ([call("bash_code_execution", {"command": cmd}, f"t{i}")], "tool_use")

    fin = ([text("done")], "end_turn")
    rounds = [
        bash("python3 -c 'print(123*456)'", 1),
        fin,
        bash("echo streamed-ok", 2),
        fin,
        bash("echo persisted > note.txt && ls", 3),
        fin,
        bash("cat note.txt", 4),
        fin,
        bash('printf "a,b\\n1,2\\n" > "$OUTPUT_DIR/out.csv" && ls "$OUTPUT_DIR"', 5),
        fin,
    ]
    c, _ = real(rounds)
    c.headers.update({"authorization": f"Bearer {mod.TOKEN}"})
    cl = anthropic.Anthropic(
        base_url="http://testserver", api_key=mod.TOKEN, http_client=c, max_retries=0
    )
    res: dict = {}
    mod.basic("http://testserver", res, cl=cl)
    assert all(v["ok"] for v in res.values()) and set(res) == {
        "plain",
        "stream",
        "container_reuse",
        "file_output",
    }
    # sha suite
    words = ["alpha", "beta", "gamma", "delta"]
    script = (
        "python3 - <<'EOF'\nimport hashlib\nrows=['name,sha256']+[w+','+hashlib.sha256(w.encode()).hexdigest() for w in WORDS]\n"
        "open('hashes.csv','w').write('\\n'.join(rows)+'\\n')\nEOF\ncp hashes.csv \"$OUTPUT_DIR\"/ && ls \"$OUTPUT_DIR\""
    ).replace("WORDS", repr(words))
    c2, _ = real([bash(script, 9), ([text("alpha row")], "end_turn")])
    c2.headers.update({"authorization": f"Bearer {mod.TOKEN}"})
    cl2 = anthropic.Anthropic(
        base_url="http://testserver", api_key=mod.TOKEN, http_client=c2, max_retries=0
    )
    res2: dict = {}
    mod.sha("http://testserver", res2, cl=cl2)
    assert res2["sha256_csv"]["ok"] and res2["sha256_csv"]["detail"]["rows"] == 4
    assert hashlib.sha256(b"alpha").hexdigest()

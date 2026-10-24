"""API parity probes: does each request field reach the engine on every dialect, streamed and not?

``api_matrix.json`` cells that cite a ``probe`` are executed here against the scripted engine
(``wire_harness``): the request goes through the real router over ``TestClient`` and the engine
call it produced is checked. A cell is therefore ``engaged`` only while its probe passes; a probe
that fails belongs to a ``partial`` / ``missing`` cell, never to a green one.

CPU only; no model, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .wire_clients import Clients, run
from .wire_harness import Script, install

MATRIX = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "python"
        / "yunshu_gateway"
        / "api_matrix.json"
    ).read_text()
)

USER = [{"role": "user", "content": "hi"}]
BASE = {
    "chat": ("/v1/chat/completions", {"model": "m", "messages": USER}),
    "responses": ("/v1/responses", {"model": "m", "input": "hi"}),
    "messages": (
        "/v1/messages",
        {"model": "m", "max_tokens": 50, "messages": USER},
    ),
    "completions": ("/v1/completions", {"model": "m", "prompt": "hi"}),
}
ALL = tuple(BASE)
SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}


def same(frag: dict, kw: dict, cols=ALL) -> dict:
    return {c: (frag, kw) for c in cols}


# probe name -> {column: (request fragment, expected engine kwargs)}
PROBES: dict[str, dict[str, tuple[dict, dict]]] = {
    "temperature": same({"temperature": 0.3}, {"temperature": 0.3}),
    "top_p": same({"top_p": 0.8}, {"top_p": 0.8}),
    "top_k": same({"top_k": 7}, {"top_k": 7}),
    "min_p": same({"min_p": 0.1}, {"min_p": 0.1}),
    "repetition_penalty": same(
        {"repetition_penalty": 1.1}, {"repetition_penalty": 1.1}
    ),
    "frequency_penalty": same({"frequency_penalty": 0.3}, {"frequency_penalty": 0.3}),
    "presence_penalty": same({"presence_penalty": 0.2}, {"presence_penalty": 0.2}),
    "logit_bias": same({"logit_bias": {"5": 1.0}}, {"logit_bias": {5: 1.0}}),
    "seed": same({"seed": 5}, {"seed": 5}),
    "stop_token_ids": same({"stop_token_ids": [3]}, {"stop_token_ids": [3]}),
    "min_tokens": same({"min_tokens": 2}, {"min_tokens": 2}),
    "ignore_eos": same({"ignore_eos": True}, {"ignore_eos": True}),
    "suppress_tokens": same({"suppress_tokens": [9]}, {"suppress_tokens": [9]}),
    "xtc": same(
        {"xtc_probability": 0.5, "xtc_threshold": 0.2},
        {"xtc_probability": 0.5, "xtc_threshold": 0.2},
    ),
    "top_n_sigma": same({"top_n_sigma": 1.5}, {"top_n_sigma": 1.5}),
    "spec_decode": same({"spec_decode": True}, {"spec_decode": True}),
    "priority": same({"priority": 2}, {"priority": 2}),
    "timeout": same({"timeout": 30}, {"timeout_seconds": 30}),
    "max_tokens": {
        "chat": ({"max_tokens": 7}, {"max_tokens": 7}),
        "responses": ({"max_output_tokens": 7}, {"max_tokens": 7}),
        "messages": ({"max_tokens": 7}, {"max_tokens": 7}),
        "completions": ({"max_tokens": 7}, {"max_tokens": 7}),
    },
    "stop": {
        "chat": ({"stop": ["zz"]}, {"stop": ["zz"]}),
        "responses": ({"stop": ["zz"]}, {"stop": ["zz"]}),
        "messages": ({"stop_sequences": ["zz"]}, {"stop": ["zz"]}),
        "completions": ({"stop": ["zz"]}, {"stop": ["zz"]}),
    },
    "logprobs": {
        "chat": (
            {"logprobs": True, "top_logprobs": 2},
            {"logprobs": True, "top_logprobs": 2},
        ),
        "responses": (
            {"top_logprobs": 2, "logprobs": True},
            {"logprobs": True, "top_logprobs": 2},
        ),
        "messages": (
            {"logprobs": True, "top_logprobs": 2},
            {"logprobs": True, "top_logprobs": 2},
        ),
        "completions": ({"logprobs": 2}, {"logprobs": True, "top_logprobs": 2}),
    },
    "reasoning_effort": same(
        {"reasoning_effort": "low"},
        {"reasoning_effort": "low"},
        ("chat", "completions", "responses", "messages"),
    ),
    "thinking_budget": {
        "chat": (
            {"thinking_budget": 77, "enable_thinking": True},
            {"thinking_budget": 77, "enable_thinking": True},
        ),
        "responses": (
            {"thinking_budget": 77, "enable_thinking": True},
            {"thinking_budget": 77, "enable_thinking": True},
        ),
        "messages": (
            {"max_tokens": 200, "thinking": {"type": "enabled", "budget_tokens": 77}},
            {"thinking_budget": 77, "enable_thinking": True},
        ),
        "completions": (
            {"thinking_budget": 77, "enable_thinking": True},
            {"thinking_budget": 77, "enable_thinking": True},
        ),
    },
    "lora_adapter": same({"lora_adapter": "x"}, {"lora_adapter": "x"}),
    "structured_output": {
        "chat": (
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "s", "schema": SCHEMA},
                }
            },
            {"json_schema": ...},
        ),
        "responses": (
            {
                "text": {
                    "format": {"type": "json_schema", "name": "s", "schema": SCHEMA}
                }
            },
            {"json_schema": ...},
        ),
        "messages": (
            {"output_config": {"format": {"type": "json_schema", "schema": SCHEMA}}},
            {"json_schema": ...},
        ),
        "completions": (
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "s", "schema": SCHEMA},
                }
            },
            {"json_schema": ...},
        ),
    },
    "json_object": {
        "chat": (
            {"response_format": {"type": "json_object"}},
            {"json_schema": "json_object"},
        ),
        "responses": (
            {"text": {"format": {"type": "json_object"}}},
            {"json_schema": "json_object"},
        ),
        "completions": (
            {"response_format": {"type": "json_object"}},
            {"json_schema": "json_object"},
        ),
    },
    "reasoning_object": {
        "responses": ({"reasoning": {"effort": "low"}}, {"reasoning_effort": "low"}),
        "background": ({"reasoning": {"effort": "low"}}, {"reasoning_effort": "low"}),
    },
    "grammar_regex": same(
        {"grammar": {"type": "regex", "pattern": "a+"}},
        {"json_schema": {"type": "regex", "pattern": "a+"}},
    ),
    "guided_regex": {
        c: ({"guided_regex": "a+"}, {"json_schema": {"type": "regex", "pattern": "a+"}})
        for c in ("chat", "completions")
    },
    "logprobs_include": {
        "responses": (
            {"include": ["message.output_text.logprobs"], "top_logprobs": 2},
            {"logprobs": True, "top_logprobs": 2},
        ),
        "background": (
            {"include": ["message.output_text.logprobs"], "top_logprobs": 2},
            {"logprobs": True, "top_logprobs": 2},
        ),
    },
    "system_cache_control": {
        "messages": (
            {
                "system": [
                    {
                        "type": "text",
                        "text": "sys",
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            },
            {"kv_cache_breakpoints": ...},
        ),
    },
}
# Fields that also ride a background response (the stored request runs the same handler).
BACKGROUND_FWD = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "seed",
    "stop_token_ids",
    "min_tokens",
    "ignore_eos",
    "suppress_tokens",
    "xtc",
    "top_n_sigma",
    "spec_decode",
    "priority",
    "timeout",
    "max_tokens",
    "stop",
    "logprobs",
    "reasoning_effort",
    "thinking_budget",
    "lora_adapter",
    "structured_output",
    "json_object",
    "grammar_regex",
)
for _n in BACKGROUND_FWD:
    if "responses" in PROBES[_n]:
        PROBES[_n]["background"] = PROBES[_n]["responses"]


def _post(c, col, frag, stream):
    path, base = BASE[col]
    return c.post(path, json={**base, **frag, "stream": stream})


THINK = [
    ("<think>", "reasoning"),
    ("think", "reasoning"),
    ("ing", "reasoning"),
    ("</think>", "reasoning"),
]


def _call(city: str) -> str:
    body = json.dumps({"name": "get_weather", "arguments": {"city": city}})
    return f"<tool_call>\n{body}\n</tool_call>"


def _need(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _usage(o, eng):
    _need((o.prompt, o.completion) == (7, 3), f"usage {o.prompt}/{o.completion}")


def _cached(o, eng):
    _need(o.cached == 4 and o.prompt == 9, f"cached {o.cached} prompt {o.prompt}")


def _reasoning(o, eng):
    # The scripted engine labels the tag pieces "reasoning" too; whether the tag text is
    # stripped from a streamed delta depends on the real engine's state machine, so only the
    # split itself is asserted.
    _need("thinking" in o.thinking, f"reasoning {o.thinking!r}")
    _need("<" not in o.text, f"reasoning leaked into text {o.text!r}")
    _need(o.text.strip() == "Hello there!", f"text {o.text!r}")


def _length(o, eng):
    _need(o.finish == "length", f"finish {o.finish}")
    _need(o.text.strip() == "abc", f"text {o.text!r}")


def _stop(o, eng):
    _need(
        o.finish in ("stop", "stop_sequence") and "END" not in o.text,
        f"{o.finish} {o.text!r}",
    )
    _need(o.text.strip() == "Hello world", f"text {o.text!r}")


def _tool(o, eng):
    _need(o.tools == [("get_weather", {"city": "Paris"})], f"tools {o.tools}")
    _need(o.finish == "tool_calls", f"finish {o.finish}")


def _parallel(o, eng):
    _need(len(o.tools) == 2, f"tools {o.tools}")


def _parallel_off(o, eng):
    _need(o.tools == [("get_weather", {"city": "Paris"})], f"tools {o.tools}")


def _schema(o, eng):
    _need(json.loads(o.text) == {"a": 3}, f"text {o.text!r}")
    _need(
        any(c.get("json_schema") for c in eng.calls), "json_schema never reached engine"
    )


def _none(o, eng):
    _need(o.tools == [], f"tools {o.tools}")


OA = ("chat", "responses", "messages")
# name -> (columns, Script kwargs, run() kwargs, check)
BEHAVIORS = {
    "usage_counts": (
        ALL,
        dict(pieces=["Hello", " there", "!"], prompt_tokens=7),
        {},
        _usage,
    ),
    "cached_tokens": (
        ALL,
        dict(pieces=["Hello", " there", "!"], prompt_tokens=9, cached_tokens=4),
        {},
        _cached,
    ),
    "reasoning_split": (
        OA,
        dict(pieces=[*THINK, "Hello", " there", "!"], prompt_tokens=7),
        {},
        _reasoning,
    ),
    "finish_length": (
        ALL,
        dict(pieces=["a", "b", "c", "d", "e", "f"], prompt_tokens=5),
        dict(max_tokens=3),
        _length,
    ),
    "stop_string": (
        ALL,
        dict(pieces=["Hello", " wor", "ld", " END", " more"], prompt_tokens=5),
        dict(stop=["END"]),
        _stop,
    ),
    "tool_call": (OA, dict(pieces=[_call("Paris")]), dict(tools=True), _tool),
    "tool_parallel": (
        OA,
        dict(pieces=[_call("Paris"), "\n", _call("Rome")]),
        dict(tools=True),
        _parallel,
    ),
    "tool_parallel_off": (
        OA,
        dict(pieces=[_call("Paris"), "\n", _call("Rome")]),
        dict(tools=True, parallel=False),
        _parallel_off,
    ),
    "tool_choice_forced": (
        OA,
        dict(pieces=[_call("Paris")]),
        dict(tools=True, tool_choice="get_weather"),
        _tool,
    ),
    "tool_choice_none": (
        OA,
        dict(pieces=["no call"]),
        dict(tools=True, tool_choice="none"),
        _none,
    ),
    "json_schema": (
        ("chat", "responses"),
        dict(pieces=['{"a": 3}']),
        dict(schema=True),
        _schema,
    ),
}


# ── executors ────────────────────────────────────────────────────────────────────
def _split(cell: str) -> tuple[str, str]:
    col, mode = cell.split("/")
    return col, mode


def _expect(calls, kw) -> str | None:
    if not calls:
        return "no engine call"
    for k, v in kw.items():
        got = calls[0].get(k)
        if v is ...:
            if not got:
                return f"engine got {k}={got!r}, want a non-empty value"
        elif got != v:
            return f"engine got {k}={got!r}, want {v!r}"
    return None


def _poll_background(c, rid, timeout=8.0):
    import time

    end = time.time() + timeout
    g = {}
    while time.time() < end:
        g = c.get(f"/v1/responses/{rid}").json()
        if g.get("status") not in ("queued", "in_progress"):
            return g
        time.sleep(0.05)
    return g


def _fwd(name):
    def check(mp, cell):
        col, mode = _split(cell)
        if mode == "background":
            col = "background"
        frag, kw = PROBES[name][col]
        c, eng = install(mp, Script())
        if mode == "background":
            with c:
                r = c.post(
                    "/v1/responses",
                    json={**BASE["responses"][1], **frag, "background": True},
                )
                if r.status_code != 200:
                    return f"HTTP {r.status_code}"
                g = _poll_background(c, r.json()["id"])
            if g.get("status") != "completed":
                return f"background ended {g.get('status')}"
            return _expect(eng.calls, kw)
        r = _post(c, col, frag, mode == "stream")
        if r.status_code != 200:
            return f"HTTP {r.status_code}"
        return _expect(eng.calls, kw)

    return check


class _BgResponses:
    """``client.responses`` with every create turned into background + poll (non-stream shape)."""

    def __init__(self, real):
        self.real = real

    def create(self, **args):
        import time

        r = self.real.create(background=True, **args)
        assert r.status in ("queued", "in_progress"), r.status
        end = time.time() + 8
        while time.time() < end:
            g = self.real.retrieve(r.id)
            if g.status not in ("queued", "in_progress"):
                return g
            time.sleep(0.05)
        return g


def _beh(name):
    def check(mp, cell):
        col, mode = _split(cell)
        _cols, script, kw, fn = BEHAVIORS[name]
        c, eng = install(mp, Script(**script))
        try:
            cl = Clients(c)
            if mode == "background":
                cl.oa.responses = _BgResponses(cl.oa.responses)
                with c:
                    o = run(cl, "responses", stream=False, **kw)
            else:
                o = run(cl, col, stream=mode == "stream", **kw)
            fn(o, eng)
        except AssertionError as e:
            return str(e)[:160]
        except Exception as e:  # noqa: BLE001 - a crash is a failed probe
            return f"{type(e).__name__}: {e}"[:160]
        return None

    return check


def _bg_run(c, body):
    r = c.post(
        "/v1/responses", json={**BASE["responses"][1], **body, "background": True}
    )
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    if r.json().get("status") not in ("queued", "in_progress"):
        return None, f"first answer was {r.json().get('status')}"
    return _poll_background(c, r.json()["id"]), None


def _bg_usage(mp, cell):
    c, eng = install(mp, Script(pieces=["Hello", " there", "!"], prompt_tokens=7))
    with c:
        g, why = _bg_run(c, {})
    if why:
        return why
    u = g.get("usage") or {}
    return (
        None
        if (u.get("input_tokens"), u.get("output_tokens")) == (7, 3)
        else f"usage {u}"
    )


def _bg_tool(mp, cell):
    c, eng = install(mp, Script(pieces=[_call("Paris")]))
    with c:
        g, why = _bg_run(
            c,
            {
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                ]
            },
        )
    if why:
        return why
    calls = [o for o in g.get("output", []) if o.get("type") == "function_call"]
    if len(calls) == 1 and json.loads(calls[0]["arguments"]) == {"city": "Paris"}:
        return None
    return f"output {g.get('output')}"


def _bg_text(mp, cell):
    c, eng = install(mp, Script(pieces=["a", "b", "c"]))
    with c:
        g, why = _bg_run(c, {})
    if why:
        return why
    try:
        text = g["output"][-1]["content"][0]["text"]
    except Exception:  # noqa: BLE001
        return f"output {g.get('output')}"
    return (
        None
        if text == "abc" and g["status"] == "completed"
        else f"{g['status']} {text!r}"
    )


def _bg_stream(mp, cell):
    c, eng = install(mp, Script(pieces=["a", "b", "c"]))
    with c:
        r = c.post(
            "/v1/responses",
            json={**BASE["responses"][1], "background": True, "stream": True},
        )
        if r.status_code != 200 or "response.created" not in r.text:
            return f"HTTP {r.status_code} {r.text[:80]!r}"
        if "response.completed" not in r.text:
            return "stream never completed"
    return None


def _bg_cancel(mp, cell):
    c, eng = install(mp, Script(pieces=["a"] * 40, delay=0.05))
    with c:
        r = c.post("/v1/responses", json={**BASE["responses"][1], "background": True})
        rid = r.json()["id"]
        cr = c.post(f"/v1/responses/{rid}/cancel")
        if cr.status_code != 200 or cr.json().get("status") != "cancelled":
            return f"cancel -> {cr.status_code} {cr.text[:80]}"
        g = _poll_background(c, rid, 3.0)
    return None if g.get("status") == "cancelled" else f"poll says {g.get('status')}"


def _store(mp, cell):
    col, mode = _split(cell)
    c, eng = install(mp, Script(pieces=["a", "b"]))
    with c:
        if col == "chat":
            r = c.post(
                "/v1/chat/completions",
                json={**BASE["chat"][1], "store": True, "stream": mode == "stream"},
            )
            if r.status_code != 200:
                return f"HTTP {r.status_code}"
            if mode == "stream":
                rid = next(
                    json.loads(ln[6:])["id"]
                    for ln in r.text.split("\n")
                    if ln.startswith("data: {")
                )
            else:
                rid = r.json()["id"]
            base = f"/v1/chat/completions/{rid}"
        else:
            r = c.post(
                "/v1/responses",
                json={
                    **BASE["responses"][1],
                    "store": True,
                    "stream": mode == "stream",
                },
            )
            if r.status_code != 200:
                return f"HTTP {r.status_code}"
            if mode == "stream":
                rid = json.loads(
                    next(
                        ln[6:]
                        for ln in r.text.split("\n")
                        if ln.startswith("data: {") and "response.created" in ln
                    )
                )["response"]["id"]
            else:
                rid = r.json()["id"]
            base = f"/v1/responses/{rid}"
        if c.get(base).status_code != 200:
            return "stored object not retrievable"
        if c.delete(base).status_code != 200:
            return "delete failed"
        return (
            None if c.get(base).status_code == 404 else "still retrievable after delete"
        )


def _prev(mp, cell):
    col, mode = _split(cell)
    c, eng = install(mp, Script(pieces=["a", "b", "c"]))
    with c:
        body1 = {**BASE["responses"][1], "store": True}
        r1 = c.post("/v1/responses", json=body1)
        if r1.status_code != 200:
            return f"HTTP {r1.status_code}"
        rid = r1.json()["id"]
        eng.calls.clear()
        body = {"model": "m", "input": "again", "previous_response_id": rid}
        if mode == "background":
            g, why = _bg_run(c, body)
            if why:
                return why
        else:
            r2 = c.post("/v1/responses", json={**body, "stream": mode == "stream"})
            if r2.status_code != 200:
                return f"HTTP {r2.status_code}"
    msgs = json.dumps(eng.calls[0]["messages"]) if eng.calls else ""
    return (
        None if ("abc" in msgs and "hi" in msgs) else "previous turn not in the prompt"
    )


def _conversation(mp, cell):
    col, mode = _split(cell)
    c, eng = install(mp, Script(pieces=["a", "b", "c"]))
    with c:
        cv = c.post("/v1/conversations", json={})
        if cv.status_code != 200:
            return f"conversations HTTP {cv.status_code}"
        cid = cv.json()["id"]
        r = c.post(
            "/v1/responses",
            json={
                **BASE["responses"][1],
                "conversation": cid,
                "stream": mode == "stream",
            },
        )
        if r.status_code != 200:
            return f"HTTP {r.status_code}"
        items = c.get(f"/v1/conversations/{cid}/items").json().get("data", [])
    return None if len(items) >= 2 else f"{len(items)} items stored"


def _n_choices(mp, cell):
    col, mode = _split(cell)
    c, eng = install(mp, Script(pieces=["a", "b"]))
    path, base = BASE[col]
    r = c.post(path, json={**base, "n": 2, "stream": mode == "stream"})
    if r.status_code != 200:
        return f"HTTP {r.status_code}"
    if len(eng.calls) != 2:
        return f"{len(eng.calls)} engine calls for n=2"
    if mode == "nonstream":
        return None if len(r.json()["choices"]) == 2 else "not two choices"
    idx = {
        ch["index"]
        for ln in r.text.split("\n")
        if ln.startswith("data: {")
        for ch in json.loads(ln[6:]).get("choices", [])
    }
    return None if idx == {0, 1} else f"stream indexes {idx}"


def _usage_opt(mp, cell):
    col, mode = _split(cell)
    c, eng = install(mp, Script(pieces=["a", "b"], prompt_tokens=4))
    path, base = BASE[col]

    def usage_chunk(opts):
        r = c.post(path, json={**base, "stream": True, **opts})
        return any(
            ln.startswith("data: {") and '"usage": {' in ln and '"choices": []' in ln
            for ln in r.text.split("\n")
        )

    if usage_chunk({}):
        return "usage chunk sent without include_usage"
    if not usage_chunk({"stream_options": {"include_usage": True}}):
        return "no usage chunk with include_usage"
    return None


def _request_id(mp, cell):
    col, mode = _split(cell)
    c, eng = install(mp, Script())
    path, base = BASE[col]
    r = c.post(
        path,
        json={**base, "stream": mode == "stream"},
        headers={"X-Request-Id": "rid-42"},
    )
    return (
        None if r.headers.get("x-request-id") == "rid-42" else "X-Request-Id not echoed"
    )


def _x_yunshu(mp, cell):
    col, mode = _split(cell)
    c, eng = install(mp, Script())
    path, base = BASE[col]
    r = c.post(path, json={**base, "stream": mode == "stream"})
    if mode == "stream":
        return (
            None
            if ("x_yunshu" in r.text or "yunshu-stats" in r.text)
            else "no x_yunshu in the stream"
        )
    return None if "x_yunshu" in r.json() else "no x_yunshu object in the body"


def _error_envelope(mp, cell):
    col, mode = _split(cell)
    c, eng = install(mp, Script())
    path, base = BASE[col]
    extra = {"background": True} if mode == "background" else {}
    r = c.post(
        path, json={**base, "temperature": "hot", "stream": mode == "stream", **extra}
    )
    if r.status_code != 400:
        return f"HTTP {r.status_code}"
    j = r.json()
    if col == "messages":
        ok = j.get("type") == "error" and {"type", "message"} <= set(j.get("error", {}))
    else:
        ok = {"message", "type"} <= set(j.get("error", {}))
    return None if ok else f"envelope {str(j)[:100]}"


def _asgi_drive(app, path, body, disconnect_after):
    import asyncio
    import time

    payload = json.dumps(body).encode()

    async def go():
        first = True

        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": payload, "more_body": False}
            await asyncio.sleep(disconnect_after)
            return {"type": "http.disconnect"}

        async def send(_m):
            pass

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [
                (b"host", b"testserver"),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode()),
            ],
            "client": ("127.0.0.1", 1),
            "server": ("testserver", 80),
            "app": app,
        }
        t = time.time()
        await asyncio.wait_for(app(scope, receive, send), 15)
        return time.time() - t

    return asyncio.run(go())


def _disconnect(mp, cell):
    col, mode = _split(cell)
    full = 60 * 0.05  # seconds the scripted generation needs when nobody stops it
    c, eng = install(mp, Script(pieces=["a"] * 60, delay=0.05))
    path, base = BASE[col]
    took = _asgi_drive(c.app, path, {**base, "stream": mode == "stream"}, 0.4)
    return (
        None
        if took < full * 0.75
        else f"generation ran {took:.1f}s of {full:.1f}s after the client left"
    )


def _poll_text(mp, c, col, mode, frag):
    """The raw text of a response for one cell (stored object for background)."""
    path, base = BASE[col if col != "background" else "responses"]
    if mode == "background":
        g, why = _bg_run(c, frag)
        return None if g is None else json.dumps(g), why
    r = c.post(path, json={**base, **frag, "stream": mode == "stream"})
    return r.text, (None if r.status_code == 200 else f"HTTP {r.status_code}")


def _echo(pattern_by_col, frag_by_col):
    import re

    def check(mp, cell):
        col, mode = _split(cell)
        c, eng = install(mp, Script(pieces=["a"]))
        with c:
            text, why = _poll_text(mp, c, col, mode, frag_by_col.get(col, {}))
        if why:
            return why
        return (
            None
            if re.search(pattern_by_col[col], text)
            else f"{pattern_by_col[col]} not in the response"
        )

    return check


def _tool_result_input(mp, cell):
    col, mode = _split(cell)
    c, eng = install(mp, Script(pieces=["a"]))
    args = json.dumps({"city": "Paris"})
    frags = {
        "chat": {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": args},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "SUNNY-42"},
            ],
        },
        "responses": {
            "input": [
                {"type": "message", "role": "user", "content": "hi"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": args,
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "SUNNY-42",
                },
            ],
        },
        "messages": {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": {"city": "Paris"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "SUNNY-42",
                        }
                    ],
                },
            ],
        },
    }
    tools = {
        "chat": [
            {
                "type": "function",
                "function": {"name": "get_weather", "parameters": {"type": "object"}},
            }
        ],
        "responses": [
            {
                "type": "function",
                "name": "get_weather",
                "parameters": {"type": "object"},
            }
        ],
        "messages": [
            {
                "name": "get_weather",
                "description": "w",
                "input_schema": {"type": "object"},
            }
        ],
    }
    with c:
        frag = {**frags[col], "tools": tools[col]}
        if mode == "background":
            g, why = _bg_run(c, frag)
            if why:
                return why
        else:
            path, base = BASE[col]
            r = c.post(path, json={**base, **frag, "stream": mode == "stream"})
            if r.status_code != 200:
                return f"HTTP {r.status_code}: {r.text[:100]}"
    seen = json.dumps(eng.calls[0].get("messages")) if eng.calls else ""
    return None if "SUNNY-42" in seen else "the tool result never reached the prompt"


def _system_prompt(mp, cell):
    col, mode = _split(cell)
    c, eng = install(mp, Script(pieces=["a"]))
    frags = {
        "chat": {
            "messages": [
                {"role": "system", "content": "SYS-7"},
                {"role": "user", "content": "hi"},
            ]
        },
        "responses": {"instructions": "SYS-7"},
        "messages": {"system": "SYS-7"},
    }
    with c:
        if mode == "background":
            g, why = _bg_run(c, frags[col])
            if why:
                return why
        else:
            path, base = BASE[col]
            r = c.post(path, json={**base, **frags[col], "stream": mode == "stream"})
            if r.status_code != 200:
                return f"HTTP {r.status_code}"
    return (
        None
        if "SYS-7" in json.dumps(eng.calls[0].get("messages"))
        else "system text not in the prompt"
    )


def _overflow(mp, cell):
    """A prompt longer than the model's context window is a 400 that names the window."""
    from yunshu_gateway import streaming

    col, mode = _split(cell)
    c, eng = install(mp, Script())
    mp.setattr(streaming, "get_max_context_window", lambda *a, **k: 20)
    long_text = "word " * 400
    frags = {
        "chat": {"messages": [{"role": "user", "content": long_text}]},
        "responses": {"input": long_text},
        "messages": {"messages": [{"role": "user", "content": long_text}]},
        "completions": {"prompt": long_text},
    }
    path, base = BASE[col]
    extra = {"background": True} if mode == "background" else {}
    with c:
        r = c.post(
            path, json={**base, **frags[col], "stream": mode == "stream", **extra}
        )
        if mode == "background":
            if r.status_code == 400:
                return None
            g = _poll_background(c, r.json()["id"])
            return (
                None
                if g.get("status") == "failed"
                else f"background ended {g.get('status')}"
            )
    if r.status_code != 400:
        return f"HTTP {r.status_code}: {r.text[:100]}"
    return None if "context window" in r.text else f"body {r.text[:100]}"


CHECKS: dict[str, dict[str, object]] = {}
for _n, _cols in PROBES.items():
    CHECKS[_n] = {}
    for col in _cols:
        if col == "background":
            CHECKS[_n]["responses/background"] = _fwd(_n)
        else:
            for m in ("nonstream", "stream"):
                CHECKS[_n][f"{col}/{m}"] = _fwd(_n)
for _n, (_cols, *_r) in BEHAVIORS.items():
    CHECKS[_n] = {
        f"{col}/{m}": _beh(_n) for col in _cols for m in ("nonstream", "stream")
    }
    if "responses" in _cols:
        CHECKS[_n]["responses/background"] = _beh(_n)
_R = ("responses/nonstream", "responses/stream", "responses/background")
_ALLNS = [f"{c}/{m}" for c in ALL for m in ("nonstream", "stream")]
CHECKS.update(
    {
        "usage_counts_background": {"responses/background": _bg_usage},
        "tool_call_background": {"responses/background": _bg_tool},
        "background_complete": {"responses/background": _bg_text},
        "background_stream": {"responses/background": _bg_stream},
        "background_cancel": {"responses/background": _bg_cancel},
        "store_retrieve_delete": {
            k: _store
            for k in (
                "chat/nonstream",
                "chat/stream",
                "responses/nonstream",
                "responses/stream",
            )
        },
        "previous_response_id": {k: _prev for k in _R},
        "conversation": {k: _conversation for k in _R[:2]},
        "n_choices": {
            k: _n_choices
            for k in (
                "chat/nonstream",
                "chat/stream",
                "completions/nonstream",
                "completions/stream",
            )
        },
        "stream_usage_option": {
            "chat/stream": _usage_opt,
            "completions/stream": _usage_opt,
        },
        "context_overflow": {k: _overflow for k in [*_ALLNS, "responses/background"]},
        "request_id": {k: _request_id for k in _ALLNS},
        "model_echo": {
            **{
                k: _echo({c: '"model": ?"m"' for c in ALL + ("background",)}, {})
                for k in _ALLNS + ["responses/background"]
            }
        },
        "service_tier_echo": {
            k: _echo(
                {
                    "chat": r'"service_tier": ?"[a-z_]+"',
                    "completions": r'"service_tier": ?"[a-z_]+"',
                    "responses": r'"service_tier": ?"[a-z_]+"',
                    "messages": r'"service_tier": ?"[a-z_]+"',
                    "background": r'"service_tier": ?"[a-z_]+"',
                },
                {c: {"service_tier": "auto"} for c in ALL + ("background",)},
            )
            for k in _ALLNS + ["responses/background"]
            if not k.startswith("completions")
        },
        "metadata_echo": {
            k: _echo(
                {
                    "chat": r'"metadata": ?\{"a": ?"b"\}',
                    "responses": r'"metadata": ?\{"a": ?"b"\}',
                    "background": r'"metadata": ?\{"a": ?"b"\}',
                },
                {
                    "chat": {"metadata": {"a": "b"}},
                    "responses": {"metadata": {"a": "b"}},
                    "background": {"metadata": {"a": "b"}},
                },
            )
            for k in ("responses/nonstream", "responses/stream", "responses/background")
        },
        "user_echo": {
            k: _echo({"responses": r'"user": ?"u"'}, {"responses": {"user": "u"}})
            for k in ("responses/nonstream", "responses/stream", "responses/background")
        },
        "tool_result_input": {
            k: _tool_result_input for k in _ALLNS[:6] + ["responses/background"]
        },
        "system_prompt": {
            k: _system_prompt for k in _ALLNS[:6] + ["responses/background"]
        },
        "x_yunshu": {k: _x_yunshu for k in _ALLNS},
        "error_envelope": {
            k: _error_envelope for k in [*_ALLNS, "responses/background"]
        },
        "disconnect": {k: _disconnect for k in _ALLNS},
    }
)


def _states(name, cell):
    """(row, status, primary) for every row whose cell cites this probe: as its own probe
    (primary: a partial cell's probe must fail) or as one of the row's extra probes (secondary:
    only an engaged cell must pass it)."""
    out = []
    for fid, feat in MATRIX["features"].items():
        if name not in feat.get("probes", []):
            continue
        c = feat["cells"].get(cell)
        if c is None:
            continue
        own = c.get("probe")
        out.append((fid, c["s"], own == name))
    return out


def _all_checks():
    for name, cells in CHECKS.items():
        for cell in cells:
            yield pytest.param(name, cell, id=f"{name}:{cell}")


@pytest.mark.parametrize("name,cell", list(_all_checks()))
def test_probe(monkeypatch, name, cell):
    """An engaged cell passes its probes; a partial / missing cell's own probe must fail (else flip it)."""
    states = [s for s in _states(name, cell) if s[1] != "na"]
    if not states:
        pytest.skip("no engaged / partial cell cites this probe")
    why = CHECKS[name][cell](monkeypatch, cell)
    for fid, state, primary in states:
        if state == "engaged":
            assert why is None, (
                f"{fid} {name}/{cell}: the matrix says engaged but {why}"
            )
        elif primary:
            assert why is not None, (
                f"{fid} {name}/{cell}: the probe passes now; flip the cell to engaged"
            )

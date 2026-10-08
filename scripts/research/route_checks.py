"""Real-server checks for every route the gateway registers (the route coverage registry).

Each check drives a *running* Yunshu server through the official `openai` / `anthropic` SDKs (typed
response models validate the shapes), `httpx` for routes with no SDK (Ollama, MCP, tokenize ...) and
`websockets` for the socket routes. `tests/unit/test_route_coverage.py` requires every route of the
app to appear in a check below or in EXEMPT with a reason, so a new route without a real check fails
CI. `scripts/research/m3sweep_jobs.py routes` runs the checks (`m3sweep` job kind `routes`).

A check is `@check(name, "METHOD /path", ..., needs="main"|"multi")`; it raises `Fail` (or any
exception) on a problem and may call `skip(reason)` when the loaded model cannot serve the route
(a route verified by no job is reported as unverified and fails the job).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

REGISTRY: dict[str, Check] = {}


class Fail(AssertionError):  # noqa: N818
    pass


class Skip(Exception):  # noqa: N818
    pass


def skip(reason: str):
    raise Skip(reason)


def expect(cond, msg: str):
    if not cond:
        raise Fail(msg)


@dataclass
class Check:
    name: str
    fn: Callable[[Ctx], Any]
    routes: tuple[str, ...]
    needs: str = "main"
    served: bool = True


def check(name: str, *routes: str, served: bool, needs: str = "main"):
    def deco(fn):
        expect(name not in REGISTRY, f"duplicate check {name}")
        REGISTRY[name] = Check(name, fn, routes, needs, served)
        return fn

    return deco


# Routes that have no real-server check, each with the reason. Keep this list short: a route
# belongs here only when no server run (with the four small checkpoints) can exercise it.
EXEMPT: dict[str, str] = {
    "POST /api/push": "documented 501 by design: nothing to push to",
    "POST /v1/decisions": "needs a decision checkpoint (Cloudflare Clef); verified by the decisions-* gpuq jobs (scripts/research/decisions_verify.py), unit-tested with a fake engine and the openai 3.26 client",
    "POST /v1/systemone": "same engine and checkpoint as /v1/decisions (TypeSafe Jev wire); see decisions_verify.py",
}


def checked_routes() -> set[str]:
    """Routes any check touches (served or not)."""
    return {r for c in REGISTRY.values() for r in c.routes}


def served_routes(exclude_needs: tuple[str, ...] = ()) -> set[str]:
    """Routes with at least one SERVED check: a model that has the capability, a successful
    response, validated content. An absent-capability / error-path check never counts."""
    return {
        r
        for c in REGISTRY.values()
        if c.served and c.needs not in exclude_needs
        for r in c.routes
    }


# ── context ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Ctx:
    url: str
    token: str
    model: str
    kind: str  # "vlm" (Qwen3.5 runner) or "text" (mlx-lm fast path) or "multi"
    http: Any = None  # httpx.Client with the bearer token
    oa: Any = None
    an: Any = None
    notes: dict = field(default_factory=dict)
    mm_models: list = field(default_factory=list)  # multi-model server: model ids
    log_tail: Any = None  # callable(n) -> the server log's last n lines
    shared: dict = field(
        default_factory=dict
    )  # kept across the servers of one job (TTS -> ASR)
    fake: Any = None  # FakeBackend (search / MCP / page on loopback), multi server only
    fixtures: dict = field(
        default_factory=dict
    )  # omni jobs: "speech_wav" bytes, "phrase"

    downgraded: bool = (
        False  # a served check that ran down an error path on this server
    )

    def unserved(self):
        """Call when this run of a served check could not exercise the served path (a text-only
        model refusing an image): the run then does not count as coverage."""
        self.downgraded = True

    def auth(self, extra=None):
        h = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        h.update(extra or {})
        return h

    def req(self, method, path, **kw):
        return self.http.request(
            method, path, headers=self.auth(kw.pop("headers", None)), **kw
        )

    def ws_url(self, path):
        return self.url.replace("http://", "ws://") + path

    def ws(self, path, **kw):
        from websockets.sync.client import connect

        return connect(
            self.ws_url(path),
            additional_headers=self.auth(),
            open_timeout=30,
            close_timeout=10,
            max_size=None,
            **kw,
        )


def jbody(r) -> Any:
    try:
        return r.json()
    except ValueError:
        return r.text


def err_ok(r, family: str):
    """Fail unless the response is an error in the family's shape with a non-empty message."""
    b = jbody(r)
    expect(
        r.status_code >= 400, f"status {r.status_code}, want an error: {str(b)[:120]}"
    )
    if family == "ollama":
        expect(
            isinstance(b, dict) and isinstance(b.get("error"), str) and b["error"],
            f"not the Ollama error shape: {str(b)[:120]}",
        )
    elif family == "anthropic":
        e = b.get("error") if isinstance(b, dict) else None
        expect(
            isinstance(b, dict)
            and b.get("type") == "error"
            and isinstance(e, dict)
            and e.get("type")
            and e.get("message"),
            f"not the Anthropic error shape: {str(b)[:120]}",
        )
    else:
        e = b.get("error") if isinstance(b, dict) else None
        expect(
            isinstance(e, dict) and e.get("message") and e.get("type"),
            f"not the OpenAI error shape: {str(b)[:120]}",
        )
    return b


def ok_or_absent(ctx: Ctx, method, path, family="openai", **kw):
    """The route either serves (2xx) or answers the family's error shape (4xx/5xx with a message,
    never a bare 500 traceback or an empty body). Returns (status, body)."""
    r = ctx.req(method, path, **kw)
    if r.status_code < 400:
        ctx.notes[f"{method} {path}"] = f"{r.status_code} served"
        return r.status_code, jbody(r)
    b = err_ok(r, family)
    if r.status_code == 500 and ctx.log_tail:
        why = [
            ln
            for ln in ctx.log_tail(200).splitlines()
            if "ERROR" in ln or "Error" in ln or "Traceback" in ln
        ][-6:]
        expect(
            False,
            f"{method} {path}: bare 500 {str(b)[:100]} | server log: {' // '.join(why)[:900]}",
        )
    e = b.get("error")
    msg = e.get("message") if isinstance(e, dict) else e
    ctx.notes[f"{method} {path}"] = f"{r.status_code}: {str(msg)[:110]}"
    return r.status_code, b


# ── docs / health / operations ───────────────────────────────────────────────────────────


@check(
    "docs",
    "GET /openapi.json",
    "GET /docs",
    "GET /docs/oauth2-redirect",
    "GET /redoc",
    served=True,
)
def _docs(c: Ctx):
    o = c.http.get("/openapi.json")
    expect(o.status_code == 200, f"openapi {o.status_code}")
    spec = o.json()
    expect("/v1/chat/completions" in spec.get("paths", {}), "openapi without chat path")
    for p in ("/docs", "/docs/oauth2-redirect", "/redoc"):
        r = c.http.get(p)
        expect(
            r.status_code == 200 and "html" in r.headers.get("content-type", ""),
            f"{p} {r.status_code}",
        )


@check(
    "health",
    "GET /health",
    "GET /health/ready",
    "GET /health/live",
    "GET /version",
    served=True,
)
def _health(c: Ctx):
    for p in ("/health", "/health/ready", "/health/live", "/version"):
        r = c.http.get(p)  # unauthenticated on purpose: health is exempt from the token
        expect(r.status_code == 200, f"{p} -> {r.status_code} {r.text[:80]}")
        expect(isinstance(r.json(), dict), f"{p} not a JSON object")
    expect(c.http.get("/health/ready").json().get("ready") is True, "ready is not true")
    expect(c.http.get("/version").json().get("version"), "version empty")


@check("metrics", "GET /metrics", served=True)
def _metrics(c: Ctx):
    r = c.req("GET", "/metrics")
    expect(r.status_code == 200, f"{r.status_code} {r.text[:80]}")
    expect(re.search(r"^# (HELP|TYPE) ", r.text, re.M), "no prometheus HELP/TYPE lines")


@check("auth", "GET /v1/models", needs="multi", served=False)
def _auth(c: Ctx):
    expect(c.token, "auth check needs the token server")
    r = c.http.get("/v1/models")
    err_ok(r, "openai")
    expect(r.status_code == 401, f"no token -> {r.status_code}, want 401")
    r = c.http.get("/v1/models", headers={"x-api-key": c.token})
    expect(r.status_code == 200, f"x-api-key token -> {r.status_code}")


@check(
    "operations",
    "GET /v1/yunshu/status",
    "GET /v1/active-generations",
    "GET /v1/requests",
    "GET /v1/requests/{request_id}",
    "DELETE /v1/requests/{request_id}",
    "POST /v1/cancel",
    "POST /v1/yunshu/warmup",
    served=True,
)
def _operations(c: Ctx):
    s = c.req("GET", "/v1/yunshu/status")
    expect(
        s.status_code == 200 and s.json().get("object") == "yunshu.status",
        f"status {s.text[:100]}",
    )
    a = c.req("GET", "/v1/active-generations")
    expect(
        a.status_code == 200 and isinstance(a.json(), dict), f"active {a.status_code}"
    )
    q = c.req("GET", "/v1/requests")
    expect(
        q.status_code == 200 and q.json().get("object") == "list",
        f"requests {q.text[:100]}",
    )
    r = c.req("GET", "/v1/requests/does-not-exist")
    err_ok(r, "openai")
    expect(r.status_code == 404, f"unknown request -> {r.status_code}")
    r = c.req("DELETE", "/v1/requests/does-not-exist")
    expect(r.status_code == 404, f"cancel unknown request -> {r.status_code}")
    r = c.req("POST", "/v1/cancel", json={"request_id": "does-not-exist"})
    expect(
        r.status_code in (200, 404), f"cancel by id -> {r.status_code} {r.text[:100]}"
    )
    w = c.req(
        "POST", "/v1/yunshu/warmup", json={"model": c.model, "prompt": "Hello, world."}
    )
    expect(w.status_code == 200, f"warmup {w.status_code} {w.text[:200]}")


@check(
    "cancel_live",
    "POST /v1/cancel",
    "DELETE /v1/requests/{request_id}",
    "GET /v1/requests/{request_id}",
    served=True,
)
def _cancel_live(c: Ctx):
    """Cancel a real in-flight stream by its X-Request-Id, both routes; the stream ends early and
    the request disappears from the active list."""
    import threading

    for how in ("cancel", "delete"):
        rid = f"routes-cancel-{how}-{int(time.time())}"
        res: dict = {}

        def run():
            try:
                with c.http.stream(
                    "POST",
                    "/v1/chat/completions",
                    headers=c.auth({"X-Request-Id": rid}),
                    json={
                        "model": c.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": "Count from 1 to 3000, one per line.",
                            }
                        ],
                        "max_tokens": 3000,
                        "stream": True,
                    },
                    timeout=120,
                ) as r:
                    res["status"] = r.status_code
                    n = 0
                    for line in r.iter_lines():
                        if line.startswith("data:"):
                            n += 1
                            res["chunks"] = n
            except Exception as e:  # noqa: BLE001
                res["err"] = repr(e)

        t = threading.Thread(target=run)
        t.start()
        deadline = time.time() + 60
        seen = False
        while time.time() < deadline and not seen:
            time.sleep(0.5)
            g = c.req("GET", f"/v1/requests/{rid}")
            seen = g.status_code == 200
        expect(seen, f"request {rid} never visible in /v1/requests/{{id}}")
        if how == "cancel":
            r = c.req("POST", "/v1/cancel", json={"request_id": rid})
        else:
            r = c.req("DELETE", f"/v1/requests/{rid}")
        expect(r.status_code == 200, f"{how} -> {r.status_code} {r.text[:100]}")
        t.join(60)
        expect(not t.is_alive(), f"stream still running 60 s after {how}")
        expect(res.get("chunks", 0) < 2900, f"stream was not stopped by {how}: {res}")
        time.sleep(1)
        expect(
            c.req("GET", f"/v1/requests/{rid}").status_code == 404,
            "cancelled request still listed",
        )


# ── generation routes (SDK, typed models) ────────────────────────────────────────────────


@check("chat", "POST /v1/chat/completions", served=True)
def _chat(c: Ctx):
    r = c.oa.chat.completions.create(
        model=c.model, messages=[{"role": "user", "content": "Say hi."}], max_tokens=16
    )
    expect(r.choices and r.usage and r.usage.prompt_tokens > 0, "chat shape")
    s = c.oa.chat.completions.create(
        model=c.model,
        messages=[{"role": "user", "content": "Say hi."}],
        max_tokens=16,
        stream=True,
        stream_options={"include_usage": True},
    )
    last = None
    for ch in s:
        last = ch
    expect(last is not None and last.usage is not None, "stream usage chunk missing")


@check("completions", "POST /v1/completions", served=True)
def _completions(c: Ctx):
    r = c.oa.completions.create(
        model=c.model, prompt="The capital of France is", max_tokens=8
    )
    expect(r.choices and r.usage and r.usage.prompt_tokens > 0, "completions shape")


@check("messages", "POST /v1/messages", "POST /messages", served=True)
def _messages(c: Ctx):
    r = c.an.messages.create(
        model=c.model, max_tokens=16, messages=[{"role": "user", "content": "Say hi."}]
    )
    expect(
        r.content is not None and r.usage.input_tokens > 0 and r.stop_reason,
        "messages shape",
    )
    # the unprefixed alias, raw (the SDK always uses /v1)
    rr = c.req(
        "POST",
        "/messages",
        headers={"anthropic-version": "2023-06-01"},
        json={
            "model": c.model,
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    expect(
        rr.status_code == 200 and rr.json().get("type") == "message",
        f"/messages {rr.status_code}",
    )


@check("responses_basic", "POST /v1/responses", served=True)
def _responses_basic(c: Ctx):
    r = c.oa.responses.create(model=c.model, input="Say hi.", max_output_tokens=32)
    expect(
        r.status in ("completed", "incomplete") and r.usage.input_tokens > 0,
        "responses shape",
    )
    s = c.oa.responses.create(
        model=c.model, input="Say hi.", max_output_tokens=32, stream=True
    )
    types = [e.type for e in s]
    expect(
        "response.created" in types
        and types[-1] in ("response.completed", "response.incomplete"),
        f"events {types[:3]}..{types[-1:]}",
    )


@check("ollama_generate", "POST /api/chat", "POST /api/generate", served=True)
def _ollama_gen(c: Ctx):
    r = c.req(
        "POST",
        "/api/chat",
        json={
            "model": c.model,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
            "options": {"num_predict": 8},
        },
    )
    b = r.json()
    expect(
        r.status_code == 200
        and b.get("done") is True
        and "message" in b
        and b.get("prompt_eval_count", 0) > 0,
        f"chat {str(b)[:150]}",
    )
    r = c.req(
        "POST",
        "/api/generate",
        json={
            "model": c.model,
            "prompt": "Hi",
            "stream": False,
            "options": {"num_predict": 8},
        },
    )
    b = r.json()
    expect(
        r.status_code == 200 and b.get("done") is True and "response" in b,
        f"generate {str(b)[:150]}",
    )
    with c.http.stream(
        "POST",
        "/api/generate",
        headers=c.auth(),
        json={"model": c.model, "prompt": "Hi", "options": {"num_predict": 8}},
    ) as s:
        lines = [json.loads(x) for x in s.iter_lines() if x.strip()]
    expect(
        lines and lines[-1].get("done") is True and all("response" in x for x in lines),
        "ollama NDJSON stream",
    )


# ── models ───────────────────────────────────────────────────────────────────────────────


@check("models", "GET /v1/models", "GET /v1/models/{model_id:path}", served=True)
def _models(c: Ctx):
    ml = c.oa.models.list()
    expect(ml.data and ml.data[0].id, "openai models.list empty")
    m = c.oa.models.retrieve(ml.data[0].id)
    expect(m.id == ml.data[0].id and m.object == "model", "openai models.retrieve")
    am = c.an.models.list()
    expect(am.data and am.data[0].display_name, "anthropic models.list ModelInfo")
    a1 = c.an.models.retrieve(am.data[0].id)
    expect(a1.id == am.data[0].id and a1.type == "model", "anthropic models.retrieve")
    r = c.req("GET", "/v1/models/no-such-model-xyz")
    expect(
        r.status_code in (200, 404), f"unknown model -> {r.status_code}"
    )  # single-model mode: advisory


@check(
    "models_multi",
    "POST /v1/models/load",
    "POST /v1/models/unload/{model_id:path}",
    needs="multi",
    served=True,
)
def _models_multi(c: Ctx):
    expect(len(c.mm_models) >= 2, f"multi-model server lists {c.mm_models}")
    a, b = c.mm_models[:2]
    ids = [m.id for m in c.oa.models.list().data]
    expect(a in ids and b in ids, f"models.list {ids} lacks {a} / {b}")
    r = c.req("POST", "/v1/models/load", json={"model": a}, timeout=300)
    expect(
        r.status_code == 200 and r.json().get("status") == "loaded",
        f"load {r.status_code} {r.text[:150]}",
    )
    ch = c.oa.chat.completions.create(
        model=a, messages=[{"role": "user", "content": "Hi"}], max_tokens=8
    )
    expect(ch.choices, "chat on loaded model")
    r = c.req("POST", "/v1/models/unload/" + a)
    expect(r.status_code == 200, f"unload {r.status_code} {r.text[:150]}")
    r = c.req("POST", "/v1/models/unload/" + a)
    err_ok(
        r, "openai"
    )  # already unloaded / not loaded: an error with a message, not a 500
    expect(r.status_code != 500, f"second unload 500 {r.text[:100]}")
    r = c.req("POST", "/v1/models/load", json={"model": "no-such-model"})
    err_ok(r, "openai")
    expect(r.status_code == 404, f"load unknown -> {r.status_code}")
    r = c.req("POST", "/v1/models/load", json={"model": ""})
    expect(r.status_code == 400, f"load empty -> {r.status_code}")
    # reload by chat (auto-load) still works after an explicit unload
    ch = c.oa.chat.completions.create(
        model=a, messages=[{"role": "user", "content": "Hi"}], max_tokens=8
    )
    expect(ch.choices, "chat after unload (auto reload)")


@check("models_admin_denied", "POST /v1/models/load", served=False)
def _models_denied(c: Ctx):
    r = c.http.post("/v1/models/load", json={"model": c.model})  # no token
    err_ok(r, "openai")
    expect(r.status_code == 401, f"load without token -> {r.status_code}")


# ── files, batches (OpenAI and Anthropic) ────────────────────────────────────────────────


def _batch_lines(c: Ctx, n=2, endpoint="/v1/chat/completions"):
    out = []
    for i in range(n):
        out.append(
            json.dumps(
                {
                    "custom_id": f"req-{i}",
                    "method": "POST",
                    "url": endpoint,
                    "body": {
                        "model": c.model,
                        "messages": [
                            {"role": "user", "content": f"Reply with the number {i}."}
                        ],
                        "max_tokens": 12,
                    },
                }
            )
        )
    return ("\n".join(out) + "\n").encode()


@check(
    "files",
    "POST /v1/files",
    "GET /v1/files",
    "GET /v1/files/{file_id}",
    "DELETE /v1/files/{file_id}",
    "GET /v1/files/{file_id}/content",
    served=True,
)
def _files(c: Ctx):
    data = b'{"a": 1}\n{"a": 2}\n'
    f = c.oa.files.create(file=("data.jsonl", data), purpose="batch")
    expect(
        f.id
        and f.object == "file"
        and f.bytes == len(data)
        and f.filename == "data.jsonl",
        f"create {f}",
    )
    expect(f.id in [x.id for x in c.oa.files.list().data], "file not in list")
    g = c.oa.files.retrieve(f.id)
    expect(g.id == f.id and g.purpose == "batch", "retrieve")
    expect(c.oa.files.content(f.id).read() == data, "content bytes differ")
    d = c.oa.files.delete(f.id)
    expect(d.deleted is True and d.id == f.id, f"delete {d}")
    r = c.req("GET", f"/v1/files/{f.id}")
    err_ok(r, "openai")
    expect(r.status_code == 404, f"retrieve after delete -> {r.status_code}")
    r = c.req("DELETE", f"/v1/files/{f.id}")
    expect(r.status_code == 404, f"second delete -> {r.status_code}")
    # Anthropic Files shape (beta files-api)
    af = c.an.beta.files.upload(
        file=("note.txt", b"hello anthropic files", "text/plain")
    )
    expect(
        af.id and af.filename == "note.txt" and af.size_bytes == 21,
        f"anthropic upload {af}",
    )
    expect(af.id in [x.id for x in c.an.beta.files.list().data], "anthropic list")
    md = c.an.beta.files.retrieve_metadata(af.id)
    expect(md.id == af.id, "anthropic metadata")
    # like the hosted API, only files created by tools can be downloaded, not uploaded ones
    try:
        c.an.beta.files.download(af.id)
        raise Fail("download of an uploaded file succeeded")
    except Fail:
        raise
    except Exception as ex:  # noqa: BLE001
        expect(
            getattr(ex, "status_code", None) == 403
            and "cannot be downloaded" in str(ex),
            f"anthropic download of an uploaded file: {type(ex).__name__} {str(ex)[:120]}",
        )
    dd = c.an.beta.files.delete(af.id)
    expect(dd.id == af.id, f"anthropic delete {dd}")
    r = c.req("GET", f"/v1/files/{af.id}", headers={"anthropic-version": "2023-06-01"})
    err_ok(r, "anthropic")
    expect(r.status_code == 404, f"anthropic retrieve after delete -> {r.status_code}")
    # file uploaded then used by a message document block
    af2 = c.an.beta.files.upload(
        file=("fact.txt", b"The secret word is PINEAPPLE.", "text/plain")
    )
    try:
        m = c.an.beta.messages.create(
            model=c.model,
            max_tokens=24,
            betas=["files-api-2025-04-14"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "file", "file_id": af2.id},
                        },
                        {"type": "text", "text": "What is the secret word?"},
                    ],
                }
            ],
        )
        expect(
            m.usage.input_tokens > 8,
            f"document not inlined: input_tokens {m.usage.input_tokens}",
        )
    finally:
        c.an.beta.files.delete(af2.id)


@check(
    "batches",
    "POST /v1/batches",
    "GET /v1/batches",
    "GET /v1/batches/{batch_id}",
    "POST /v1/batches/{batch_id}/cancel",
    served=True,
)
def _batches(c: Ctx):
    inp = c.oa.files.create(file=("batch.jsonl", _batch_lines(c, 2)), purpose="batch")
    b = c.oa.batches.create(
        input_file_id=inp.id, endpoint="/v1/chat/completions", completion_window="24h"
    )
    expect(b.id and b.object == "batch" and b.input_file_id == inp.id, f"create {b}")
    expect(b.id in [x.id for x in c.oa.batches.list().data], "not listed")
    deadline = time.time() + 600
    cur = b
    while time.time() < deadline and cur.status not in (
        "completed",
        "failed",
        "cancelled",
        "expired",
    ):
        time.sleep(3)
        cur = c.oa.batches.retrieve(b.id)
    expect(cur.status == "completed", f"batch status {cur.status} errors={cur.errors}")
    rc = cur.request_counts
    expect(rc.total == 2 and rc.completed == 2 and rc.failed == 0, f"counts {rc}")
    expect(cur.output_file_id, "no output_file_id")
    rows = [
        json.loads(x)
        for x in c.oa.files.content(cur.output_file_id).text.splitlines()
        if x.strip()
    ]
    expect(
        sorted(r["custom_id"] for r in rows) == ["req-0", "req-1"], f"output ids {rows}"
    )
    for r in rows:
        body = r["response"]["body"]
        expect(
            r["response"]["status_code"] == 200 and body["choices"][0]["message"],
            f"row {r}",
        )
    # invalid input: bad endpoint / wrong file are 4xx with a message
    bad = c.req(
        "POST",
        "/v1/batches",
        json={
            "input_file_id": "file-nope",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    err_ok(bad, "openai")
    expect(bad.status_code in (400, 404), f"missing input file -> {bad.status_code}")
    # cancel an in-flight batch
    big = c.oa.files.create(file=("big.jsonl", _batch_lines(c, 24)), purpose="batch")
    b2 = c.oa.batches.create(
        input_file_id=big.id, endpoint="/v1/chat/completions", completion_window="24h"
    )
    cc = c.oa.batches.cancel(b2.id)
    expect(cc.status in ("cancelling", "cancelled"), f"cancel -> {cc.status}")
    t0 = time.time()
    while time.time() - t0 < 240 and c.oa.batches.retrieve(b2.id).status != "cancelled":
        time.sleep(3)
    expect(
        c.oa.batches.retrieve(b2.id).status == "cancelled",
        "batch never reached cancelled",
    )
    # cancel of a finished batch is an error, not a 500
    r = c.req("POST", f"/v1/batches/{b.id}/cancel")
    expect(r.status_code in (200, 400, 409), f"cancel finished -> {r.status_code}")
    for fid in (inp.id, big.id, cur.output_file_id):
        c.oa.files.delete(fid)


@check(
    "messages_batches",
    "POST /v1/messages/batches",
    "GET /v1/messages/batches",
    "GET /v1/messages/batches/{batch_id}",
    "GET /v1/messages/batches/{batch_id}/results",
    "POST /v1/messages/batches/{batch_id}/cancel",
    "DELETE /v1/messages/batches/{batch_id}",
    served=True,
)
def _messages_batches(c: Ctx):
    def reqs(n):
        return [
            {
                "custom_id": f"m-{i}",
                "params": {
                    "model": c.model,
                    "max_tokens": 12,
                    "messages": [
                        {"role": "user", "content": f"Reply with the number {i}."}
                    ],
                },
            }
            for i in range(n)
        ]

    b = c.an.messages.batches.create(requests=reqs(2))
    expect(b.id and b.processing_status in ("in_progress", "ended"), f"create {b}")
    expect(b.id in [x.id for x in c.an.messages.batches.list().data], "not listed")
    deadline = time.time() + 600
    cur = b
    while time.time() < deadline and cur.processing_status != "ended":
        time.sleep(3)
        cur = c.an.messages.batches.retrieve(b.id)
    expect(cur.processing_status == "ended", f"status {cur.processing_status}")
    expect(
        cur.request_counts.succeeded == 2 and cur.request_counts.errored == 0,
        f"counts {cur.request_counts}",
    )
    res = list(c.an.messages.batches.results(b.id))
    expect(sorted(r.custom_id for r in res) == ["m-0", "m-1"], f"results {res}")
    for r in res:
        expect(
            r.result.type == "succeeded" and r.result.message.content is not None,
            f"result {r}",
        )
    b2 = c.an.messages.batches.create(requests=reqs(24))
    cc = c.an.messages.batches.cancel(b2.id)
    expect(
        cc.processing_status in ("canceling", "ended"),
        f"cancel -> {cc.processing_status}",
    )
    t0 = time.time()
    while (
        time.time() - t0 < 240
        and c.an.messages.batches.retrieve(b2.id).processing_status != "ended"
    ):
        time.sleep(3)
    end2 = c.an.messages.batches.retrieve(b2.id)
    expect(
        end2.processing_status == "ended" and end2.cancel_initiated_at is not None,
        f"canceled batch {end2}",
    )
    for bid in (b.id, b2.id):
        d = c.an.messages.batches.delete(bid)
        expect(d.id == bid and d.type == "message_batch_deleted", f"delete {d}")
    r = c.req(
        "GET",
        f"/v1/messages/batches/{b.id}",
        headers={"anthropic-version": "2023-06-01"},
    )
    err_ok(r, "anthropic")
    expect(r.status_code == 404, f"retrieve after delete -> {r.status_code}")


# ── conversations, responses lifecycle ───────────────────────────────────────────────────


@check(
    "conversations",
    "POST /v1/conversations",
    "GET /v1/conversations/{conversation_id}",
    "POST /v1/conversations/{conversation_id}",
    "DELETE /v1/conversations/{conversation_id}",
    "POST /v1/conversations/{conversation_id}/items",
    "GET /v1/conversations/{conversation_id}/items",
    "GET /v1/conversations/{conversation_id}/items/{item_id}",
    "DELETE /v1/conversations/{conversation_id}/items/{item_id}",
    served=True,
)
def _conversations(c: Ctx):
    cv = c.oa.conversations.create(
        metadata={"topic": "routes"},
        items=[{"type": "message", "role": "user", "content": "My name is Ada."}],
    )
    expect(
        cv.id and cv.object == "conversation" and cv.metadata == {"topic": "routes"},
        f"create {cv}",
    )
    items = c.oa.conversations.items.list(cv.id)
    expect(
        len(items.data) == 1 and items.data[0].type == "message",
        f"initial items {items}",
    )
    up = c.oa.conversations.update(cv.id, metadata={"topic": "updated"})
    expect(up.metadata == {"topic": "updated"}, f"update {up}")
    added = c.oa.conversations.items.create(
        cv.id, items=[{"type": "message", "role": "user", "content": "I like tea."}]
    )
    expect(len(added.data) == 1 and added.data[0].id, f"items.create {added}")
    one = c.oa.conversations.items.retrieve(added.data[0].id, conversation_id=cv.id)
    expect(one.id == added.data[0].id, "items.retrieve")
    # generation against the conversation appends the turn
    r = c.oa.responses.create(
        model=c.model,
        conversation=cv.id,
        input="What is my name?",
        max_output_tokens=48,
    )
    expect(
        r.conversation and r.conversation.id == cv.id,
        f"response.conversation {r.conversation}",
    )
    after = c.oa.conversations.items.list(cv.id)
    types = [i.type for i in after.data]
    expect(len(after.data) >= 4, f"conversation did not grow: {types}")
    dl = c.oa.conversations.items.delete(added.data[0].id, conversation_id=cv.id)
    expect(dl.id == cv.id, f"item delete returns the conversation: {dl}")
    gone = c.req("GET", f"/v1/conversations/{cv.id}/items/{added.data[0].id}")
    err_ok(gone, "openai")
    expect(gone.status_code == 404, f"deleted item -> {gone.status_code}")
    d = c.oa.conversations.delete(cv.id)
    expect(d.deleted is True and d.id == cv.id, f"delete {d}")
    r2 = c.req("GET", f"/v1/conversations/{cv.id}")
    err_ok(r2, "openai")
    expect(r2.status_code == 404, f"retrieve deleted -> {r2.status_code}")
    r3 = c.req(
        "POST",
        f"/v1/conversations/{cv.id}/items",
        json={"items": [{"type": "message", "role": "user", "content": "x"}]},
    )
    expect(r3.status_code == 404, f"items on deleted conversation -> {r3.status_code}")


@check(
    "chat_stored_completions",
    "GET /v1/chat/completions",
    "GET /v1/chat/completions/{completion_id}",
    "POST /v1/chat/completions/{completion_id}",
    "DELETE /v1/chat/completions/{completion_id}",
    "GET /v1/chat/completions/{completion_id}/messages",
    served=True,
)
def _chat_stored(c: Ctx):
    tag = f"routes-{int(time.time())}"
    msgs = [{"role": "user", "content": "Say hi in three words."}]
    r = c.oa.chat.completions.create(
        model=c.model, messages=msgs, max_tokens=24, store=True, metadata={"tag": tag}
    )
    got = c.oa.chat.completions.retrieve(r.id)
    expect(got.id == r.id and got.metadata == {"tag": tag}, f"retrieve {got.id} {got.metadata}")
    expect(
        got.choices[0].message.content == r.choices[0].message.content
        and got.usage.total_tokens == r.usage.total_tokens,
        "stored completion differs from the one returned",
    )
    stream = c.oa.chat.completions.create(
        model=c.model,
        messages=msgs,
        max_tokens=24,
        stream=True,
        store=True,
        metadata={"tag": tag},
    )
    text, sid = "", None
    for ch in stream:
        sid = ch.id
        if ch.choices and ch.choices[0].delta.content:
            text += ch.choices[0].delta.content
    sgot = c.oa.chat.completions.retrieve(sid)
    expect(
        (sgot.choices[0].message.content or "") == text,
        f"stored stream {sgot.choices[0].message.content!r} != streamed {text!r}",
    )
    listed = [x.id for x in c.oa.chat.completions.list(metadata={"tag": tag}, order="asc")]
    expect(listed == [r.id, sid], f"list by metadata {listed}")
    pm = list(c.oa.chat.completions.messages.list(r.id))
    expect(
        len(pm) == 1 and pm[0].role == "user" and pm[0].content == msgs[0]["content"],
        f"messages {pm}",
    )
    up = c.oa.chat.completions.update(r.id, metadata={"tag": tag, "x": "1"})
    expect(up.metadata == {"tag": tag, "x": "1"}, f"update {up.metadata}")
    plain = c.oa.chat.completions.create(model=c.model, messages=msgs, max_tokens=8)
    miss = c.req("GET", f"/v1/chat/completions/{plain.id}")
    err_ok(miss, "openai")
    expect(miss.status_code == 404, f"unstored completion -> {miss.status_code}")
    for cid in (r.id, sid):
        d = c.oa.chat.completions.delete(cid)
        expect(d.deleted is True and d.id == cid, f"delete {d}")
    gone = c.req("GET", f"/v1/chat/completions/{r.id}")
    expect(gone.status_code == 404, f"deleted -> {gone.status_code}")


@check(
    "realtime_client_secrets",
    "POST /v1/realtime/client_secrets",
    "POST /v1/realtime/sessions",
    "POST /v1/realtime/transcription_sessions",
    served=True,
)
def _realtime_secrets(c: Ctx):
    from websockets.sync.client import connect

    sec = c.oa.realtime.client_secrets.create(
        expires_after={"anchor": "created_at", "seconds": 300},
        session={
            "type": "realtime",
            "model": c.model,
            "instructions": "Reply briefly.",
            "output_modalities": ["text"],
        },
    )
    expect(sec.value.startswith("ek_") and sec.session.type == "realtime", f"secret {sec}")
    expect(sec.session.instructions == "Reply briefly.", "session config not echoed")
    # the ephemeral secret, not the static key, opens the socket and carries its session
    with connect(
        c.ws_url("/v1/realtime") + f"?model={c.model}",
        additional_headers={"Authorization": f"Bearer {sec.value}"},
        open_timeout=30,
        max_size=None,
    ) as ws:
        first = json.loads(ws.recv(timeout=30))
        expect(first.get("type") == "session.created", f"first event {first.get('type')}")
        expect(first["session"]["id"] == sec.session.id, "session id differs from the secret's")
        expect(first["session"]["instructions"] == "Reply briefly.", "secret session not applied")
        ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Say hi."}],
                    },
                }
            )
        )
        ws.send(json.dumps({"type": "response.create"}))
        evs = _ws_events(ws, lambda e: e.get("type") in ("response.done", "error"), timeout=240)
        expect(evs[-1]["type"] == "response.done", f"secret turn ended with {evs[-1]['type']}")
    if c.token:
        bad = None
        try:
            with connect(
                c.ws_url("/v1/realtime") + f"?model={c.model}",
                additional_headers={"Authorization": "Bearer ek_not-a-real-secret"},
                open_timeout=30,
            ) as ws2:
                bad = ws2.recv(timeout=5)
        except Exception:
            bad = None
        expect(bad is None, f"an unknown ek_ secret was accepted: {bad}")
    s = c.oa.beta.realtime.sessions.create(model=c.model, instructions="hi", modalities=["text"])
    expect(s.client_secret.value.startswith("ek_") and s.modalities == ["text"], f"sessions {s}")
    t = c.oa.beta.realtime.transcription_sessions.create(
        input_audio_transcription={"model": "whisper-1"}
    )
    expect(t.client_secret.value.startswith("ek_"), f"transcription_sessions {t}")
    bad_ttl = c.req(
        "POST",
        "/v1/realtime/client_secrets",
        json={"expires_after": {"anchor": "created_at", "seconds": 1}},
    )
    err_ok(bad_ttl, "openai")
    expect(bad_ttl.status_code == 400, f"ttl 1 -> {bad_ttl.status_code}")


@check(
    "responses_lifecycle",
    "POST /v1/responses",
    "GET /v1/responses/{response_id}",
    "DELETE /v1/responses/{response_id}",
    "POST /v1/responses/{response_id}/cancel",
    "GET /v1/responses/{response_id}/input_items",
    served=True,
)
def _responses_lifecycle(c: Ctx):
    r = c.oa.responses.create(
        model=c.model, input="Remember the number 7.", store=True, max_output_tokens=32
    )
    expect(r.id and r.status in ("completed", "incomplete"), f"create {r.status}")
    g = c.oa.responses.retrieve(r.id)
    expect(
        g.id == r.id and g.output == r.output,
        "retrieve differs from the created response",
    )
    items = c.oa.responses.input_items.list(r.id)
    expect(
        len(items.data) == 1 and items.data[0].type == "message", f"input_items {items}"
    )
    # chain through previous_response_id
    r2 = c.oa.responses.create(
        model=c.model,
        input="What number?",
        previous_response_id=r.id,
        store=True,
        max_output_tokens=32,
    )
    expect(r2.previous_response_id == r.id, "previous_response_id not echoed")
    items2 = c.oa.responses.input_items.list(r2.id)
    expect(len(items2.data) >= 1, "input_items of the second response")
    # store=False is not retrievable
    r3 = c.oa.responses.create(
        model=c.model, input="Hi", store=False, max_output_tokens=8
    )
    nr = c.req("GET", f"/v1/responses/{r3.id}")
    err_ok(nr, "openai")
    expect(nr.status_code == 404, f"unstored retrieve -> {nr.status_code}")
    # cancel of a finished response is idempotent: it returns the terminal response unchanged
    cc = c.req("POST", f"/v1/responses/{r.id}/cancel")
    expect(
        cc.status_code == 200
        and cc.json().get("status") in ("completed", "incomplete"),
        f"cancel completed -> {cc.status_code} {cc.text[:100]}",
    )
    cc = c.req("POST", "/v1/responses/resp-does-not-exist/cancel")
    err_ok(cc, "openai")
    expect(cc.status_code == 404, f"cancel unknown -> {cc.status_code}")
    bg = c.oa.responses.create(
        model=c.model,
        input="Count from 1 to 2000, one number per line.",
        background=True,
        store=True,
        max_output_tokens=2000,
    )
    expect(
        bg.status in ("queued", "in_progress", "completed"),
        f"background status {bg.status}",
    )
    x = c.oa.responses.cancel(bg.id)
    expect(x.status in ("cancelled", "completed"), f"cancel -> {x.status}")
    t0 = time.time()
    cur = c.oa.responses.retrieve(bg.id)
    while time.time() - t0 < 60 and cur.status in ("queued", "in_progress"):
        time.sleep(1)
        cur = c.oa.responses.retrieve(bg.id)
    expect(
        cur.status in ("cancelled", "completed"), f"after cancel status {cur.status}"
    )
    for rid in (r.id, r2.id, bg.id):
        d = c.oa.responses.delete(rid)
        expect(d is None or getattr(d, "deleted", True), f"delete {d}")
    gone = c.req("GET", f"/v1/responses/{r.id}")
    err_ok(gone, "openai")
    expect(gone.status_code == 404, f"retrieve after delete -> {gone.status_code}")
    gone = c.req("GET", f"/v1/responses/{r.id}/input_items")
    expect(gone.status_code == 404, f"input_items after delete -> {gone.status_code}")


@check("responses_input_tokens", "POST /v1/responses/input_tokens", served=True)
def _input_tokens(c: Ctx):
    kw = dict(
        model=c.model,
        instructions="You are terse.",
        input=[{"role": "user", "content": "Tell me about the moon in one line."}],
    )
    n = c.oa.responses.input_tokens.count(**kw)
    expect(n.object == "response.input_tokens" and n.input_tokens > 0, f"count {n}")
    r = c.oa.responses.create(max_output_tokens=8, **kw)
    expect(
        r.usage.input_tokens == n.input_tokens,
        f"input_tokens {n.input_tokens} != usage {r.usage.input_tokens}",
    )
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "w",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]
    n2 = c.oa.responses.input_tokens.count(tools=tools, **kw)
    r2 = c.oa.responses.create(max_output_tokens=8, tools=tools, **kw)
    expect(
        n2.input_tokens == r2.usage.input_tokens,
        f"with tools {n2.input_tokens} != usage {r2.usage.input_tokens}",
    )
    expect(n2.input_tokens > n.input_tokens, "tools did not add input tokens")
    # a previous_response_id chain and a conversation count what the next call really uses
    r1 = c.oa.responses.create(
        model=c.model, input="Remember the number 7.", store=True, max_output_tokens=16
    )
    kw3 = dict(model=c.model, input="Which number?", previous_response_id=r1.id)
    n3 = c.oa.responses.input_tokens.count(**kw3)
    r3 = c.oa.responses.create(max_output_tokens=8, **kw3)
    expect(
        n3.input_tokens == r3.usage.input_tokens,
        f"previous_response_id: {n3.input_tokens} != usage {r3.usage.input_tokens}",
    )
    cv = c.oa.conversations.create(
        items=[{"type": "message", "role": "user", "content": "My name is Ada."}]
    )
    kw4 = dict(model=c.model, input="What is my name?", conversation=cv.id)
    n4 = c.oa.responses.input_tokens.count(**kw4)
    r4 = c.oa.responses.create(max_output_tokens=8, **kw4)
    expect(
        n4.input_tokens == r4.usage.input_tokens,
        f"conversation: {n4.input_tokens} != usage {r4.usage.input_tokens}",
    )
    for rid in (r1.id, r3.id):
        c.oa.responses.delete(rid)
    c.oa.conversations.delete(cv.id)


@check("responses_compact", "POST /v1/responses/compact", served=True)
def _compact(c: Ctx):
    long = " ".join(
        f"Fact number {i}: the item {i} costs {i * 3} coins." for i in range(40)
    )
    r = c.oa.responses.compact(
        model=c.model,
        input=[
            {"role": "user", "content": long},
            {"role": "user", "content": "Summarise."},
        ],
    )
    expect(r.object == "response.compaction" and r.output, f"compact {str(r)[:200]}")
    expect(r.usage.input_tokens > 0, "compact usage")
    types = [o.type for o in r.output]
    expect("compaction" in types, f"no compaction item in {types}")
    # the compacted output is accepted back as input
    r2 = c.oa.responses.create(
        model=c.model,
        input=[o.model_dump(exclude_none=True) for o in r.output]
        + [{"role": "user", "content": "Continue."}],
        max_output_tokens=16,
    )
    expect(
        r2.status in ("completed", "incomplete"),
        f"continue after compaction {r2.status}",
    )


def _compact_history(n_turns: int, tag: str) -> list[dict]:
    """User/assistant text turns of roughly 60 tokens each (``tag`` makes every history unique)."""
    out = []
    for i in range(n_turns):
        out.append(
            {
                "role": "user",
                "content": f"[{tag}] Question {i}: what does record {i} of ledger {tag} "
                f"say about the invoice number {i * 7919} and the shipment of {i * 31} crates?",
            }
        )
        out.append(
            {
                "role": "assistant",
                "content": f"Record {i} of ledger {tag}: invoice {i * 7919} was paid, "
                f"shipment of {i * 31} crates left the warehouse on day {i % 28 + 1}.",
            }
        )
    out.append({"role": "user", "content": "Continue with the next record."})
    return out


def _tokens(c: Ctx, msgs: list[dict]) -> tuple[int, int | None]:
    text = "\n".join(m["content"] for m in msgs)
    r = c.http.post("/v1/tokenize", json={"model": c.model, "prompt": text}).json()
    return int(r["count"]), r.get("max_model_len")


def _history_of_tokens(
    c: Ctx, target: int, tag: str
) -> tuple[list[dict], int, int | None]:
    """A history of about ``target`` tokens (sized from a measured probe)."""
    probe = _compact_history(40, tag)
    per_turn = _tokens(c, probe)[0] / 40
    msgs = _compact_history(int(target / per_turn), tag)
    n, window = _tokens(c, msgs)
    return msgs, n, window


@check("responses_compact_history", "POST /v1/responses/compact", served=True)
def _compact_history_check(c: Ctx):
    """The summary request keeps the real messages as its prefix (cached tokens ~ history length
    for a history the server just served) and a history larger than the context window still
    compacts (several passes, each under the window)."""
    import time

    # ~20K tokens, served once so its KV is warm
    msgs, n, window = _history_of_tokens(c, 20_000, f"warm{int(time.time())}")
    expect(15_000 < n < 25_000, f"history is {n} tokens, wanted ~20K")
    c.oa.chat.completions.create(model=c.model, messages=msgs, max_tokens=1)
    r = c.oa.responses.compact(model=c.model, input=msgs)
    u = r.usage
    cached = u.input_tokens_details.cached_tokens
    c.notes["compact_cached_tokens"] = f"{cached}/{u.input_tokens} (history {n})"
    expect(
        any(o.type == "compaction" for o in r.output),
        "no compaction item (warm history)",
    )
    expect(
        cached >= 0.8 * n,
        f"compaction cached only {cached} of {u.input_tokens} input tokens "
        f"(history {n}): the summary request is not reusing the history's prefix cache",
    )
    if not window:  # /tokenize does not know it: read it off the server's own refusal
        import re

        try:
            c.oa.chat.completions.create(
                model=c.model,
                messages=[{"role": "user", "content": "a " * 2_000_000}],
                max_tokens=1,
            )
        except Exception as exc:  # noqa: BLE001
            m = re.search(r"> (\d+) maximum", str(exc))
            window = int(m.group(1)) if m else None
    c.notes["compact_window"] = str(window)
    # a history over the window (only where the model's window is small enough to build one)
    if not window or window > 40_000:
        return
    big, nb, _ = _history_of_tokens(c, int(window * 1.3), f"big{int(time.time())}")
    expect(nb > window, f"over-window history is {nb} tokens, window {window}")
    r = c.oa.responses.compact(model=c.model, input=big)
    expect(
        any(o.type == "compaction" for o in r.output),
        "no compaction item (over window)",
    )
    c.notes["compact_over_window"] = (
        f"{nb} tokens, window {window}, input_tokens summed {r.usage.input_tokens}"
    )
    expect(r.usage.input_tokens > window, "over-window compaction used one call?")


# ── token counting ───────────────────────────────────────────────────────────────────────


@check(
    "count_tokens",
    "POST /v1/messages/count_tokens",
    "POST /messages/count_tokens",
    served=True,
)
def _count_tokens(c: Ctx):
    msgs = [{"role": "user", "content": "Tell me about the moon in one line."}]
    n = c.an.messages.count_tokens(
        model=c.model, messages=msgs, system="You are terse."
    )
    expect(n.input_tokens > 0, f"count {n}")
    m = c.an.messages.create(
        model=c.model, max_tokens=4, messages=msgs, system="You are terse."
    )
    expect(
        m.usage.input_tokens
        + (m.usage.cache_read_input_tokens or 0)
        + (m.usage.cache_creation_input_tokens or 0)
        == n.input_tokens,
        f"count_tokens {n.input_tokens} != usage {m.usage}",
    )
    tools = [
        {
            "name": "get_weather",
            "description": "w",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]
    n2 = c.an.messages.count_tokens(model=c.model, messages=msgs, tools=tools)
    m2 = c.an.messages.create(model=c.model, max_tokens=4, messages=msgs, tools=tools)
    tot = (
        m2.usage.input_tokens
        + (m2.usage.cache_read_input_tokens or 0)
        + (m2.usage.cache_creation_input_tokens or 0)
    )
    expect(n2.input_tokens == tot, f"with tools {n2.input_tokens} != usage {tot}")
    rr = c.req(
        "POST",
        "/messages/count_tokens",
        headers={"anthropic-version": "2023-06-01"},
        json={"model": c.model, "messages": msgs},
    )
    expect(
        rr.status_code == 200 and rr.json().get("input_tokens", 0) > 0,
        f"/messages/count_tokens {rr.status_code}",
    )
    bad = c.req(
        "POST",
        "/v1/messages/count_tokens",
        headers={"anthropic-version": "2023-06-01"},
        json={"model": c.model},
    )
    err_ok(bad, "anthropic")
    expect(
        bad.status_code == 400, f"count_tokens without messages -> {bad.status_code}"
    )


@check(
    "tokenize",
    "POST /v1/tokenize",
    "POST /v1/detokenize",
    "POST /tokenize",
    "POST /detokenize",
    served=True,
)
def _tokenize(c: Ctx):
    text = "Hello, world! 你好"
    for pre in ("/v1", ""):
        t = c.req("POST", f"{pre}/tokenize", json={"model": c.model, "prompt": text})
        expect(t.status_code == 200, f"{pre}/tokenize {t.status_code} {t.text[:100]}")
        tj = t.json()
        expect(tj["count"] == len(tj["tokens"]) and tj["count"] > 0, f"tokenize {tj}")
        d = c.req(
            "POST", f"{pre}/detokenize", json={"model": c.model, "tokens": tj["tokens"]}
        )
        expect(
            d.status_code == 200 and d.json().get("prompt") == text,
            f"detokenize roundtrip {d.text[:100]}",
        )
    msgs = [{"role": "user", "content": "Say hi."}]
    tm = c.req("POST", "/v1/tokenize", json={"model": c.model, "messages": msgs}).json()
    ch = c.oa.chat.completions.create(model=c.model, messages=msgs, max_tokens=1)
    expect(
        tm["count"] == ch.usage.prompt_tokens,
        f"tokenize(messages) {tm['count']} != usage {ch.usage.prompt_tokens}",
    )
    bad = c.req("POST", "/v1/tokenize", json={"model": c.model})
    err_ok(bad, "openai")
    expect(
        bad.status_code in (400, 422), f"tokenize without input -> {bad.status_code}"
    )


# ── embeddings, pooling, scoring (a generation model: absent capability or a valid answer) ──


@check(
    "embeddings",
    "POST /v1/embeddings",
    "POST /v1/pooling",
    "POST /v1/score",
    "POST /v1/rerank",
    "POST /v1/classify",
    served=False,
)
def _embeddings(c: Ctx):
    # the SDK types the success shape; an error must be the OpenAI error shape with a message
    try:
        e = c.oa.embeddings.create(model=c.model, input="hello")
        expect(e.data and len(e.data[0].embedding) > 0, "embeddings shape")
        c.notes["embeddings"] = "served"
    except Exception as ex:  # noqa: BLE001
        import openai

        expect(
            isinstance(ex, openai.APIStatusError) and ex.status_code in (400, 404, 501),
            f"embeddings error {type(ex).__name__} {ex}",
        )
        expect(
            isinstance(ex.body, dict) or ex.body is None or isinstance(ex.body, str),
            "error body",
        )
        expect(str(ex.message).strip() != "", "embeddings error without message")
        c.notes["embeddings"] = f"absent {ex.status_code}: {str(ex.message)[:100]}"
    r = c.req("POST", "/v1/embeddings", json={"model": c.model, "input": ""})
    expect(r.status_code >= 400, "empty input accepted")
    err_ok(r, "openai")
    expect(r.status_code == 400, f"empty embedding input -> {r.status_code}")
    for path, body in (
        ("/v1/pooling", {"model": c.model, "input": "hello"}),
        ("/v1/score", {"model": c.model, "text_1": "a cat", "text_2": "a feline"}),
        (
            "/v1/rerank",
            {"model": c.model, "query": "cat", "documents": ["a feline", "a car"]},
        ),
        (
            "/v1/classify",
            {"model": c.model, "input": "hello", "labels": ["greeting", "farewell"]},
        ),
    ):
        st, b = ok_or_absent(c, "POST", path, json=body, timeout=120)
        c.notes[path] = st


# ── modalities a chat model cannot serve: the error must say so ──────────────────────────


def _wav(seconds=0.5, rate=16000) -> bytes:
    import io
    import math
    import struct
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(
            b"".join(
                struct.pack("<h", int(6000 * math.sin(2 * math.pi * 440 * i / rate)))
                for i in range(int(seconds * rate))
            )
        )
    return buf.getvalue()


def _png(size: int = 224, rgb=(200, 30, 30)) -> bytes:
    """A solid-colour RGB PNG of size x size (no imaging library needed)."""
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return (
            struct.pack(">I", len(data))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    row = b"\x00" + bytes(rgb) * size
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row * size))
        + chunk(b"IEND", b"")
    )


@check(
    "audio_absent",
    "POST /v1/audio/speech",
    "POST /v1/audio/speech/stream",
    "POST /v1/audio/transcriptions",
    "POST /v1/audio/translations",
    "GET /v1/audio/voices",
    served=False,
)
def _audio(c: Ctx):
    st = {}
    st["voices"] = ok_or_absent(c, "GET", "/v1/audio/voices")[0]
    st["speech"] = ok_or_absent(
        c,
        "POST",
        "/v1/audio/speech",
        json={"model": c.model, "input": "Hello", "voice": "alloy"},
        timeout=120,
    )[0]
    st["speech_stream"] = ok_or_absent(
        c,
        "POST",
        "/v1/audio/speech/stream",
        json={"model": c.model, "input": "Hello", "voice": "alloy"},
        timeout=120,
    )[0]
    st["transcriptions"] = ok_or_absent(
        c,
        "POST",
        "/v1/audio/transcriptions",
        data={"model": c.model},
        files={"file": ("a.wav", _wav(), "audio/wav")},
        timeout=120,
    )[0]
    st["translations"] = ok_or_absent(
        c,
        "POST",
        "/v1/audio/translations",
        data={"model": c.model},
        files={"file": ("a.wav", _wav(), "audio/wav")},
        timeout=120,
    )[0]
    c.notes["audio"] = st
    # the answer names what is served and what cannot be done, not "model not found"
    for key, word in (
        ("POST /v1/audio/speech", "cannot synthesize speech"),
        ("POST /v1/audio/speech/stream", "cannot synthesize speech"),
        ("POST /v1/audio/transcriptions", "cannot transcribe audio"),
        ("POST /v1/audio/translations", "cannot transcribe audio"),
    ):
        expect(word in c.notes.get(key, ""), f"{key}: {c.notes.get(key)!r}")
    # a chat model must not "succeed" at speech with an empty body
    r = c.req(
        "POST",
        "/v1/audio/speech",
        json={"model": c.model, "input": "Hello", "voice": "alloy"},
        timeout=120,
    )
    if r.status_code < 400:
        expect(len(r.content) > 44, "speech 200 with an empty audio body")


@check(
    "images_absent",
    "POST /v1/images/generations",
    "POST /v1/images/generations/stream",
    "POST /v1/images/variations",
    "POST /v1/images/edits",
    "POST /v1/ocr",
    served=False,
)
def _images(c: Ctx):
    st = {}
    st["generations"] = ok_or_absent(
        c,
        "POST",
        "/v1/images/generations",
        json={"model": c.model, "prompt": "a red square", "n": 1, "size": "256x256"},
        timeout=120,
    )[0]
    st["generations_stream"] = ok_or_absent(
        c,
        "POST",
        "/v1/images/generations/stream",
        json={"model": c.model, "prompt": "a red square", "size": "256x256"},
        timeout=120,
    )[0]
    st["variations"] = ok_or_absent(
        c,
        "POST",
        "/v1/images/variations",
        data={"model": c.model},
        files={"image": ("a.png", _png(), "image/png")},
        timeout=120,
    )[0]
    st["edits"] = ok_or_absent(
        c,
        "POST",
        "/v1/images/edits",
        data={"model": c.model, "prompt": "make it blue"},
        files={"image": ("a.png", _png(), "image/png")},
        timeout=120,
    )[0]
    st["ocr"] = ok_or_absent(
        c,
        "POST",
        "/v1/ocr",
        data={"model": c.model},
        files={"file": ("a.png", _png(), "image/png")},
        timeout=120,
    )[0]
    c.notes["images"] = st
    for key in ("POST /v1/images/generations", "POST /v1/images/generations/stream"):
        expect(
            "cannot generate or edit images" in c.notes.get(key, ""),
            f"{key}: {c.notes.get(key)!r}",
        )
    if c.kind == "vlm":  # a vision model reads the image through the OCR fallback
        expect(
            st["ocr"] == 200, f"OCR on a vision model: {c.notes.get('POST /v1/ocr')}"
        )
    else:
        expect(
            st["ocr"] == 503
            and "cannot read text from images" in c.notes.get("POST /v1/ocr", ""),
            f"OCR on a text model: {c.notes.get('POST /v1/ocr')}",
        )
    # every request above is valid: a 400 means the route rejected a well-formed request
    for k, v in st.items():
        expect(v != 400, f"{k}: a well-formed request answered 400 ({c.notes})")
    # the official SDK sends multipart for edits and variations
    import openai

    for name, fn in (
        (
            "sdk_edit",
            lambda: c.oa.images.edit(
                model=c.model, image=("a.png", _png()), prompt="make it blue"
            ),
        ),
        (
            "sdk_variation",
            lambda: c.oa.images.create_variation(
                model=c.model, image=("a.png", _png())
            ),
        ),
        (
            "sdk_generate",
            lambda: c.oa.images.generate(model=c.model, prompt="a red square"),
        ),
    ):
        try:
            fn()
            c.notes[name] = "served"
        except openai.APIStatusError as ex:
            expect(
                ex.status_code == 404 and str(ex.message).strip(),
                f"{name}: {ex.status_code} {str(ex.message)[:120]}",
            )
            c.notes[name] = f"404: {str(ex.message)[:100]}"


@check("omni_absent", "POST /v1/omni/speech/stream", served=False)
def _omni(c: Ctx):
    st, _ = ok_or_absent(
        c,
        "POST",
        "/v1/omni/speech/stream",
        json={"text": "Hello"},
        timeout=120,
    )
    expect(st != 400, "a well-formed omni request answered 400")
    c.notes["omni"] = st


# ── MCP server ───────────────────────────────────────────────────────────────────────────


@check(
    "mcp",
    "POST /v1/mcp",
    "GET /v1/mcp/tools",
    "GET /v1/mcp/sse",
    served=True,
)
def _mcp(c: Ctx):
    def rpc(method, params=None, id=1):
        r = c.req(
            "POST",
            "/v1/mcp",
            json={
                "jsonrpc": "2.0",
                "id": id,
                "method": method,
                **({"params": params} if params is not None else {}),
            },
            timeout=120,
        )
        expect(r.status_code == 200, f"{method} -> {r.status_code} {r.text[:150]}")
        return r.json()

    i = rpc(
        "initialize",
        {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "routes", "version": "1"},
        },
    )
    expect(
        i.get("jsonrpc") == "2.0" and i.get("id") == 1 and "result" in i,
        f"initialize {i}",
    )
    expect(i["result"].get("serverInfo", {}).get("name"), f"serverInfo {i['result']}")
    tl = rpc("tools/list", id=2)
    tools = tl["result"]["tools"]
    expect(
        isinstance(tools, list)
        and all("name" in t and "inputSchema" in t for t in tools),
        f"tools/list {str(tl)[:200]}",
    )
    gt = c.req("GET", "/v1/mcp/tools")
    expect(
        gt.status_code == 200 and isinstance(gt.json().get("tools"), list),
        f"GET tools {gt.text[:100]}",
    )
    expect(
        {t["name"] for t in gt.json()["tools"]} == {t["name"] for t in tools},
        "GET /mcp/tools != tools/list",
    )
    # an unknown method is a JSON-RPC error object, not an HTTP error envelope
    u = c.req(
        "POST", "/v1/mcp", json={"jsonrpc": "2.0", "id": 3, "method": "nope/never"}
    )
    expect(
        u.json().get("error", {}).get("code") == -32601 and u.json().get("id") == 3,
        f"unknown method {u.text[:150]}",
    )
    # notifications get no body
    n = c.req(
        "POST",
        "/v1/mcp",
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )
    expect(
        n.status_code in (200, 202, 204)
        and not n.text.strip().startswith('{"jsonrpc"'),
        f"notification answered: {n.status_code} {n.text[:80]}",
    )
    # a tool call, when the server exposes a tool that needs no model
    if tools:
        tc = rpc("tools/call", {"name": tools[0]["name"], "arguments": {}}, id=4)
        expect(
            "result" in tc or "error" in tc,
            f"tools/call neither result nor error: {tc}",
        )
    cs = c.req("GET", "/v1/mcp/client/status")
    expect(
        cs.status_code == 200 and "enabled" in cs.json(),
        f"client status {cs.text[:100]}",
    )
    ct = c.req("GET", "/v1/mcp/client/tools")
    expect(
        ct.status_code == 200 and isinstance(ct.json(), dict),
        f"client tools {ct.text[:100]}",
    )
    with c.http.stream("GET", "/v1/mcp/sse", headers=c.auth(), timeout=30) as s:
        expect(
            s.status_code == 200
            and "text/event-stream" in s.headers.get("content-type", ""),
            f"sse {s.status_code}",
        )
        first = next(s.iter_lines())
        expect(first.startswith("event: endpoint"), f"first sse line {first!r}")


# ── Ollama management routes ─────────────────────────────────────────────────────────────


@check(
    "ollama_native",
    "GET /api/tags",
    "GET /api/ps",
    "POST /api/show",
    "GET /api/version",
    served=True,
)
def _ollama_native(c: Ctx):
    t = c.req("GET", "/api/tags").json()
    expect(
        t["models"]
        and all(
            m.get("name") and m.get("model") and "details" in m for m in t["models"]
        ),
        f"tags {str(t)[:200]}",
    )
    ps = c.req("GET", "/api/ps").json()
    expect(isinstance(ps.get("models"), list), f"ps {ps}")
    v = c.req("GET", "/api/version").json()
    expect(re.match(r"^\d+\.\d+", v.get("version", "")), f"version {v}")
    sh = c.req("POST", "/api/show", json={"model": c.model})
    expect(
        sh.status_code == 200
        and "capabilities" in sh.json()
        and "details" in sh.json(),
        f"show {sh.text[:150]}",
    )
    err_ok(
        c.req("POST", "/api/show", json={"model": "no-such-model-xyz"}), "ollama"
    ) if c.kind == "multi" else None
    for path, body in (
        ("/api/embed", {"model": c.model, "input": "hello"}),
        ("/api/embeddings", {"model": c.model, "prompt": "hello"}),
    ):
        ok_or_absent(c, "POST", path, family="ollama", json=body, timeout=120)


# ── websockets ───────────────────────────────────────────────────────────────────────────


def _ws_events(ws, stop, timeout=180):
    evs = []
    t0 = time.time()
    while time.time() - t0 < timeout:
        raw = ws.recv(timeout=timeout)
        ev = json.loads(raw)
        evs.append(ev)
        if stop(ev):
            return evs
    raise Fail(
        f"no stop event in {timeout}s; types {[e.get('type') for e in evs][-8:]}"
    )


def _n_active(c: Ctx) -> int:
    a = c.req("GET", "/v1/active-generations").json()
    return int(a.get("count", len(a.get("data", []))))


def _was_active(c: Ctx, what: str):
    """A generation really is in flight (so the disconnect check that follows is not vacuous)."""
    t0 = time.time()
    while time.time() - t0 < 20:
        if _n_active(c) >= 1:
            return
        time.sleep(0.3)
    raise Fail(f"{what}: no active generation visible before the disconnect")


def _no_active(c: Ctx, secs=20):
    t0 = time.time()
    while time.time() - t0 < secs:
        if _n_active(c) == 0:
            return True
        time.sleep(1)
    return False


@check("ws_responses", "WS /v1/responses", served=True)
def _ws_responses(c: Ctx):
    with c.ws("/v1/responses") as ws:
        ws.send(
            json.dumps(
                {
                    "type": "response.create",
                    "model": c.model,
                    "input": "Say hi.",
                    "max_output_tokens": 24,
                    "store": False,
                }
            )
        )
        evs = _ws_events(
            ws,
            lambda e: (
                e.get("type")
                in (
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                    "error",
                )
            ),
        )
        types = [e["type"] for e in evs]
        expect(
            types[0] == "response.created"
            and types[-1] in ("response.completed", "response.incomplete"),
            f"events {types[:2]}..{types[-1:]}",
        )
        done = evs[-1]["response"]
        expect(done["usage"]["input_tokens"] > 0, "completed event without usage")
        # a second turn on the same socket chained by previous_response_id
        ws.send(
            json.dumps(
                {
                    "type": "response.create",
                    "model": c.model,
                    "input": "And again.",
                    "previous_response_id": done["id"],
                    "max_output_tokens": 16,
                    "store": True,
                }
            )
        )
        evs2 = _ws_events(
            ws,
            lambda e: (
                e.get("type")
                in (
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                    "error",
                )
            ),
        )
        expect(
            evs2[-1]["type"] in ("response.completed", "response.incomplete"),
            f"second turn {evs2[-1]['type']}",
        )
        # a malformed message is an error event, the socket survives
        ws.send("{not json")
        er = json.loads(ws.recv(timeout=30))
        expect(er.get("type") == "error", f"malformed -> {er}")
        ws.send(
            json.dumps(
                {
                    "type": "response.create",
                    "model": c.model,
                    "input": "Count to 2000 one per line.",
                    "max_output_tokens": 2000,
                    "store": False,
                }
            )
        )
        first = json.loads(ws.recv(timeout=60))
        expect(first["type"] == "response.created", f"third {first}")
        _was_active(c, "ws /v1/responses")
    # disconnect mid-generation: the server must stop it and stay healthy
    expect(_no_active(c, 60), "generation still active 60 s after the websocket closed")
    expect(
        c.http.get("/health/ready").status_code == 200,
        "server unhealthy after ws disconnect",
    )


@check("ws_auth", "WS /v1/responses", "WS /v1/stream", needs="multi", served=False)
def _ws_auth(c: Ctx):
    """With a token configured an unauthenticated upgrade is refused, an authenticated one works."""
    from websockets.sync.client import connect

    expect(c.token, "ws_auth needs the token server")
    for path in ("/v1/responses", "/v1/stream"):
        try:
            with connect(c.ws_url(path), open_timeout=15) as w:
                try:
                    w.recv(timeout=5)
                except Exception:  # noqa: BLE001  closed by the server: refused
                    continue
                raise Fail(f"unauthenticated websocket {path} accepted")
        except Fail:
            raise
        except Exception:  # noqa: BLE001  handshake refusal is the expected path
            pass
    with c.ws("/v1/stream") as w:
        expect(
            json.loads(w.recv(timeout=30)).get("type") == "session.created",
            "authenticated /v1/stream hello",
        )


@check("ws_responses_sdk", "WS /v1/responses", served=True)
def _ws_responses_sdk(c: Ctx):
    import asyncio

    import openai

    async def go():
        cl = openai.AsyncOpenAI(
            base_url=c.url + "/v1", api_key=c.token or "x", max_retries=0
        )
        async with cl.responses.connect() as conn:
            await conn.response.create(
                model=c.model, input="Say hi.", max_output_tokens=24
            )
            types = []
            async for ev in conn:
                types.append(ev.type)
                if ev.type in (
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                    "error",
                ):
                    break
        return types

    types = asyncio.run(asyncio.wait_for(go(), 180))
    expect(
        types[0] == "response.created"
        and types[-1] in ("response.completed", "response.incomplete"),
        f"sdk ws events {types[:2]}..{types[-1:]}",
    )


@check("ws_stream", "WS /v1/stream", served=True)
def _ws_stream(c: Ctx):
    with c.ws("/v1/stream") as ws:
        first = json.loads(ws.recv(timeout=30))
        expect(
            first.get("type") == "session.created"
            and first.get("protocol") == "yunshu.stream",
            f"hello {first}",
        )
        for api, body in (
            (
                "chat.completions",
                {
                    "model": c.model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 8,
                },
            ),
            ("completions", {"model": c.model, "prompt": "Hi", "max_tokens": 8}),
            ("responses", {"model": c.model, "input": "Hi", "max_output_tokens": 16}),
            (
                "messages",
                {
                    "model": c.model,
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            ),
        ):
            rid = f"r-{api}"
            ws.send(
                json.dumps({"type": "request", "id": rid, "api": api, "body": body})
            )
            evs = _ws_events(
                ws,
                lambda e, rid=rid: (
                    e.get("type") in ("done", "error") and e.get("id") in (rid, None)
                ),
            )
            expect(
                evs[-1]["type"] == "done"
                and evs[-1]["reason"] in ("completed", "stop", "length", "ok"),
                f"{api}: {evs[-1]}",
            )
            expect(any(e.get("type") == "event" for e in evs), f"{api}: no events")
        # cancel mid-generation by id
        ws.send(
            json.dumps(
                {
                    "type": "request",
                    "id": "long",
                    "api": "chat.completions",
                    "body": {
                        "model": c.model,
                        "messages": [
                            {"role": "user", "content": "Count from 1 to 3000."}
                        ],
                        "max_tokens": 3000,
                    },
                }
            )
        )
        for _ in range(200):
            e = json.loads(ws.recv(timeout=60))
            if e.get("type") == "event":
                break
        ws.send(json.dumps({"type": "cancel", "id": "long"}))
        evs = _ws_events(
            ws, lambda e: e.get("type") == "done" and e.get("id") == "long"
        )
        expect(
            evs[-1]["reason"] in ("cancelled",), f"cancel reason {evs[-1]['reason']}"
        )
        ws.send("{broken")
        er = json.loads(ws.recv(timeout=30))
        expect(er.get("type") == "error", f"malformed {er}")
        ws.send(
            json.dumps(
                {
                    "type": "request",
                    "id": "last",
                    "api": "chat.completions",
                    "body": {
                        "model": c.model,
                        "messages": [
                            {"role": "user", "content": "Count from 1 to 3000."}
                        ],
                        "max_tokens": 3000,
                    },
                }
            )
        )
        for _ in range(200):
            e = json.loads(ws.recv(timeout=60))
            if e.get("type") == "event":
                break
        _was_active(c, "ws /v1/stream")
    expect(_no_active(c, 60), "generation still active 60 s after the websocket closed")


def _realtime_text_turn(ws):
    first = json.loads(ws.recv(timeout=30))
    expect(first.get("type") == "session.created", f"first event {first.get('type')}")
    ws.send(
        json.dumps(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "output_modalities": ["text"],
                    "instructions": "Reply briefly.",
                },
            }
        )
    )
    ws.send(
        json.dumps(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Say hi."}],
                },
            }
        )
    )
    ws.send(json.dumps({"type": "response.create"}))
    evs = _ws_events(
        ws, lambda e: e.get("type") in ("response.done", "error"), timeout=240
    )
    return first, evs


@check("ws_realtime", "WS /v1/realtime", "WS /realtime", served=True)
def _ws_realtime(c: Ctx):
    for path in ("/v1/realtime", "/realtime"):
        with c.ws(path + f"?model={c.model}") as ws:
            first, evs = _realtime_text_turn(ws)
            types = [e["type"] for e in evs]
            errs = [e for e in evs if e["type"] == "error"]
            if errs and "audio" in json.dumps(errs).lower():
                c.notes[f"realtime{path}"] = f"error event: {json.dumps(errs[0])[:160]}"
                expect(
                    errs[0].get("error", {}).get("message"),
                    "realtime error without message",
                )
            else:
                expect(not errs, f"{path}: error events {errs[:1]}")
                expect(
                    types[-1] == "response.done" and "response.created" in types,
                    f"{path}: {types}",
                )
                done = evs[-1]["response"]
                expect(
                    done.get("status") in ("completed", "incomplete"),
                    f"{path}: status {done.get('status')}",
                )
                expect(
                    done.get("usage", {}).get("input_tokens", 1) > 0,
                    f"{path}: usage {done.get('usage')}",
                )
            # disconnect mid-response
            ws.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "Count from 1 to 3000."}
                            ],
                        },
                    }
                )
            )
            ws.send(json.dumps({"type": "response.create"}))
            for _ in range(100):
                e = json.loads(ws.recv(timeout=120))
                if e["type"] in (
                    "response.output_text.delta",
                    "response.text.delta",
                    "response.content_part.added",
                    "error",
                ):
                    break
        # the realtime path is not in /v1/active-generations: prove the generation stopped by the
        # engine answering a short request promptly (it would be busy for all 3000 tokens)
        t0 = time.time()
        r = c.oa.chat.completions.create(
            model=c.model,
            messages=[{"role": "user", "content": "Say ok."}],
            max_tokens=4,
        )
        expect(r.choices, "chat after realtime disconnect")
        took = time.time() - t0
        expect(
            took < 20,
            f"{path}: engine still busy {took:.0f} s after the websocket closed",
        )
    expect(
        c.http.get("/health/ready").status_code == 200,
        "server unhealthy after realtime disconnect",
    )


# ── server-side tools (web search against a local fake provider) ─────────────────────────


@check(
    "web_search_unconfigured", "POST /v1/messages", "POST /v1/responses", served=False
)
def _web_search_unconfigured(c: Ctx):
    """With no search provider configured the server tool answers in the API's own error shape."""
    m = c.an.messages.create(
        model=c.model,
        max_tokens=256,
        tool_choice={"type": "tool", "name": "web_search"},
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
        messages=[
            {"role": "user", "content": "Search the web for the weather in Paris."}
        ],
    )
    kinds = [b.type for b in m.content]
    c.notes["web_search_unconfigured"] = kinds
    expect(
        m.stop_reason in ("end_turn", "pause_turn", "max_tokens", "tool_use"),
        f"stop_reason {m.stop_reason}",
    )
    expect(m.usage.input_tokens > 0, "usage")
    r = c.oa.responses.create(
        model=c.model,
        input="Search the web for the weather in Paris.",
        tools=[{"type": "web_search"}],
        tool_choice="required",
        max_output_tokens=256,
    )
    expect(
        r.status in ("completed", "incomplete", "failed"),
        f"responses web_search {r.status}",
    )
    c.notes["web_search_responses"] = [o.type for o in r.output]


from route_checks_media import (
    wav_info,  # noqa: E402,F401  registers the modality checks
)
from route_checks_omni import (
    _omni_audio_in,  # noqa: E402,F401  registers the omni input / speech-out checks
)
from route_checks_tools import (
    FakeBackend,  # noqa: E402,F401  registers the server-tool checks
)
from route_checks_tavily import tavily_routes  # noqa: E402,F401
from route_checks_vision import (
    _vision_input,  # noqa: E402,F401  registers the image-input check
)


@check(
    "ollama_unsupported",
    "POST /api/embed",
    "POST /api/embeddings",
    "POST /api/pull",
    "POST /api/push",
    "POST /api/create",
    "POST /api/copy",
    "DELETE /api/delete",
    served=False,
)
def _ollama_unsupported(c: Ctx):
    """Error paths only: embed on a chat model (served on the embedding model by embed_served) and
    the model-management verbs, which answer a documented 501."""
    for path, body in (
        ("/api/embed", {"model": c.model, "input": "hello"}),
        ("/api/embeddings", {"model": c.model, "prompt": "hello"}),
    ):
        ok_or_absent(c, "POST", path, family="ollama", json=body, timeout=120)
    # model management verbs: an Ollama-shaped answer or a clear refusal, never a hang or a 500
    for method, path, body in (
        ("POST", "/api/pull", {"model": "unsupported-gguf:latest", "stream": False}),
        ("POST", "/api/push", {"model": "no/such-model-xyz", "stream": False}),
        (
            "POST",
            "/api/create",
            {"model": "routes-x", "from": "no-such-base", "stream": False},
        ),
        (
            "POST",
            "/api/copy",
            {"source": "no-such-model-xyz", "destination": "routes-copy"},
        ),
        ("DELETE", "/api/delete", {"model": "no-such-model-xyz"}),
    ):
        r = c.req(method, path, json=body, timeout=120)
        expect(
            r.status_code in (400, 401, 403, 404, 501),
            f"{method} {path}: {r.status_code} {r.text[:100]}",
        )
        err_ok(r, "ollama")


from route_checks_vllm import (
    _chat_validation,  # noqa: E402,F401  registers the vLLM-derived validation checks
)


@check(
    "ollama_management",
    "POST /api/pull",
    "POST /api/copy",
    "POST /api/create",
    "DELETE /api/delete",
    needs="multi",
    served=True,
)
def _ollama_management(c: Ctx):
    source = c.mm_models[0]
    copy, created = "routes-copy-native", "routes-created-native"
    try:
        r = c.req("POST", "/api/pull", json={"model": source, "stream": False})
        expect(
            r.status_code == 200 and r.json() == {"status": "success"},
            f"pull existing: {r.text}",
        )
        r = c.req("POST", "/api/copy", json={"source": source, "destination": copy})
        expect(r.status_code == 200 and not r.content, f"copy: {r.text}")
        r = c.req("POST", "/api/show", json={"model": copy})
        expect(r.status_code == 200 and r.json().get("details"), f"show copy: {r.text}")
        r = c.req("POST", "/api/create", json={"model": created, "from": source})
        expect(
            r.status_code == 200 and r.json() == {"status": "success"},
            f"create: {r.text}",
        )
        tagged = c.req(
            "POST",
            "/api/chat",
            json={
                "model": copy + ":latest",
                "stream": False,
                "messages": [{"role": "user", "content": "Hi"}],
                "options": {"num_predict": 4},
            },
        )
        expect(
            tagged.status_code == 200,
            f"latest alias: {tagged.status_code} {tagged.text[:100]}",
        )
        response = c.oa.chat.completions.create(
            model=copy, messages=[{"role": "user", "content": "Hi"}], max_tokens=4
        )
        expect(response.choices, "copied model cannot serve")
    finally:
        for name in (copy, created):
            r = c.req("DELETE", "/api/delete", json={"model": name}, timeout=300)
            expect(
                r.status_code == 200 and not r.content,
                f"delete {name}: {r.status_code} {r.text}",
            )
    unknown = c.req(
        "DELETE", "/api/delete", json={"model": "not-a-registered-model-xyz"}
    )
    expect(
        unknown.status_code == 404,
        f"delete unknown: {unknown.status_code} {unknown.text}",
    )
    ids = [m["name"] for m in c.req("GET", "/api/tags").json()["models"]]
    expect(
        copy not in ids and created not in ids and source in ids, f"delete names: {ids}"
    )


def unload_when_idle(c, model, wait=10.0):
    """Unload a model, retrying a 409 conflict for up to `wait` seconds. A socket's model lease
    is released when the handler returns, a few ms after the client sees its close handshake
    finish, so an unload sent at once can meet the still-held lease; the 409 says "retry after
    in-flight requests complete", which is what this does."""
    deadline = time.time() + wait
    while True:
        r = c.req("POST", "/v1/models/unload/" + model, timeout=300)
        if r.status_code != 409 or time.time() >= deadline:
            return r
        time.sleep(0.1)


@check(
    "realtime_lazy_load", "WS /v1/realtime", "WS /realtime", needs="multi", served=True
)
def _realtime_lazy_load(c: Ctx):
    model = c.mm_models[0]
    for path in ("/v1/realtime", "/realtime"):
        # Load then unload so every socket proves lazy loading from an unloaded state.
        r = c.req("POST", "/v1/models/load", json={"model": model}, timeout=300)
        expect(r.status_code == 200, f"load before unload: {r.text}")
        r = unload_when_idle(c, model)
        expect(r.status_code == 200, f"unload before realtime: {r.text}")
        expect(
            model not in [m["name"] for m in c.req("GET", "/api/ps").json()["models"]],
            "ps lists unloaded model",
        )
        with c.ws(path + f"?model={model}") as ws:
            _, events = _realtime_text_turn(ws)
            expect(
                not any(e["type"] == "error" for e in events),
                f"lazy realtime: {events[-1]}",
            )
            expect(
                events[-1]["type"] == "response.done", "lazy realtime did not finish"
            )
        expect(
            model in [m["name"] for m in c.req("GET", "/api/ps").json()["models"]],
            "ps omits realtime-loaded model",
        )


@check(
    "forced_tool_uncompilable",
    "POST /v1/chat/completions",
    "POST /v1/messages",
    "POST /v1/responses",
    served=False,
)
def _forced_tool_uncompilable(c: Ctx):
    if c.kind != "text":
        return
    schema = {
        "type": "object",
        "$defs": {
            "node": {"type": "object", "properties": {"next": {"$ref": "#/$defs/node"}}}
        },
        "properties": {"root": {"$ref": "#/$defs/node"}},
    }
    tools = [
        {"type": "function", "function": {"name": "recursive", "parameters": schema}}
    ]
    for stream in (False, True):
        for path, body in (
            (
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": "Call recursive"}],
                    "tools": tools,
                    "tool_choice": "required",
                },
            ),
            (
                "/v1/messages",
                {
                    "messages": [{"role": "user", "content": "Call recursive"}],
                    "max_tokens": 8,
                    "tools": [{"name": "recursive", "input_schema": schema}],
                    "tool_choice": {"type": "any"},
                },
            ),
            (
                "/v1/responses",
                {
                    "input": "Call recursive",
                    "tools": [
                        {"type": "function", "name": "recursive", "parameters": schema}
                    ],
                    "tool_choice": "required",
                },
            ),
        ):
            r = c.req("POST", path, json={"model": c.model, "stream": stream, **body})
            expect(
                r.status_code == 400
                and "Cannot guarantee forced tool_choice" in r.text,
                f"forced compilation {path}: {r.status_code} {r.text[:160]}",
            )


@check("responses_custom_tool", "POST /v1/responses", served=True)
def _responses_custom_tool(c: Ctx):
    tools = [
        {
            "type": "custom",
            "name": "echo_freeform",
            "description": "Return the requested text as freeform input",
            "format": {"type": "text"},
        }
    ]
    for stream in (False, True):
        response = c.oa.responses.create(
            model=c.model,
            input="Call echo_freeform with input HELLO.",
            tools=tools,
            tool_choice={"type": "custom", "name": "echo_freeform"},
            extra_body={"enable_thinking": False},
            max_output_tokens=256,
            stream=stream,
        )
        if stream:
            events = list(response)
            response = events[-1].response
            expect(
                any(e.type == "response.custom_tool_call_input.done" for e in events),
                "custom input done missing",
            )
        calls = [item for item in response.output if item.type == "custom_tool_call"]
        expect(
            calls
            and calls[0].name == "echo_freeform"
            and isinstance(calls[0].input, str)
            and calls[0].input,
            "custom tool output missing",
        )
        follow = c.oa.responses.create(
            model=c.model,
            previous_response_id=response.id,
            input=[
                {
                    "type": "custom_tool_call_output",
                    "call_id": calls[0].call_id,
                    "output": "HELLO",
                }
            ],
            max_output_tokens=8,
        )
        expect(follow.output, "custom tool history did not round-trip")

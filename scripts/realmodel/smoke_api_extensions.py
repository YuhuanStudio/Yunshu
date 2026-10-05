"""Real-server smoke test of the Yunshu API extensions with the official SDKs.

Starts `uvicorn yunshu_gateway.main:app` on a private port with a small model, then checks that
the OpenAI, Anthropic and Ollama SDKs keep working and that request ids, prefill progress
comments, x_yunshu stats, queue headers, /v1/requests, cancel-by-id, warmup and error hints
show up. Run it through the GPU queue with the SDK venv:

    scripts/dev/gpuq run --priority 1 --timeout 5 --stall 2 --label api-ext-smoke -- \
        $SDK_VENV/bin/python \
        scripts/realmodel/smoke_api_extensions.py

The server is always killed with SIGKILL on exit.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
PY = os.environ.get(
    "YUNSHU_SMOKE_SERVER_PY",
    str(ROOT / ".venv/bin/python"),
)
MODEL = os.environ.get(
    "YUNSHU_SMOKE_MODEL",
    str(Path("~/.yunshu/models/Qwen3.5-0.8B-MLX-bf16").expanduser()),
)
PORT = 18991
BASE = f"http://127.0.0.1:{PORT}"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(
        ("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""),
        flush=True,
    )


def start_server() -> subprocess.Popen:
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(ROOT / "python"),
        YUNSHU_MODEL=MODEL,
        YUNSHU_AUTH_DISABLED="true",
        YUNSHU_PROGRESS_INTERVAL_S="0.25",
    )
    return subprocess.Popen(
        [
            PY,
            "-m",
            "uvicorn",
            "yunshu_gateway.main:app",
            "--port",
            str(PORT),
            "--log-level",
            "warning",
        ],
        cwd=ROOT,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def wait_ready(proc: subprocess.Popen, timeout: float = 150.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("server exited during startup")
        try:
            if httpx.get(f"{BASE}/health/ready", timeout=2).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("server did not become ready")


def long_prompt(n_words: int) -> str:
    return " ".join(f"word{i % 977}" for i in range(n_words))


def main() -> int:
    proc = start_server()
    try:
        wait_ready(proc)
        run_checks()
    finally:
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait()
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


def run_checks() -> None:
    import anthropic
    import ollama
    import openai

    oa = openai.OpenAI(base_url=f"{BASE}/v1", api_key="x", max_retries=0)
    served = oa.models.list().data[0].id

    # 1. OpenAI SDK, non-streaming: spec fields + x_yunshu + headers + request id.
    raw = oa.chat.completions.with_raw_response.create(
        model=served,
        messages=[{"role": "user", "content": "Say hi in three words."}],
        max_tokens=24,
        temperature=0,
        extra_headers={"X-Request-Id": "smoke-nonstream-1"},
    )
    resp = raw.parse()
    xy = (resp.model_extra or {}).get("x_yunshu") or {}
    check("openai non-stream parses", bool(resp.choices[0].message.content is not None))
    check(
        "openai non-stream echoes X-Request-Id",
        raw.headers.get("x-request-id") == "smoke-nonstream-1",
    )
    check(
        "x_yunshu present with timings",
        xy.get("ttft_ms") is not None and xy.get("decode_tps") is not None,
        json.dumps(xy)[:300],
    )
    check(
        "engine-side prefill timings (fast path and runner)",
        xy.get("prefill_ms") is not None and xy.get("queue_wait_ms") is not None,
        json.dumps(
            {k: xy.get(k) for k in ("queue_wait_ms", "prefill_ms", "prefill_tps")}
        ),
    )
    check(
        "X-Yunshu-* headers",
        raw.headers.get("x-yunshu-decode-tps") is not None
        and raw.headers.get("x-yunshu-queue-position") == "0",
        str({k: v for k, v in raw.headers.items() if k.startswith("x-yunshu")}),
    )
    check(
        "usage untouched", resp.usage is not None and resp.usage.completion_tokens > 0
    )

    if os.environ.get("YUNSHU_SMOKE_SPEC"):
        spec = xy.get("speculative") or {}
        check(
            "speculative stats on an MTP model",
            spec.get("mode") in ("mtp", "dflash"),
            json.dumps(spec),
        )
        check(
            "speculative drafted / accepted / acceptance_rate are filled",
            bool(spec.get("drafted"))
            and spec.get("accepted") is not None
            and spec.get("acceptance_rate") is not None,
            json.dumps(spec),
        )

    # 2. OpenAI SDK streaming with include_usage: SDK ignores the comments, last chunk carries x_yunshu.
    stream = oa.chat.completions.create(
        model=served,
        messages=[{"role": "user", "content": "Count to five."}],
        max_tokens=32,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks = list(stream)
    last = chunks[-1]
    check("openai stream parses", len(chunks) > 2 and chunks[-1].usage is not None)
    check(
        "stream usage chunk has x_yunshu",
        bool((last.model_extra or {}).get("x_yunshu", {}).get("ttft_ms")),
    )

    # 3. Raw SSE with a long prompt: progress comments (queued/prefill, percent, ETA).
    body = {
        "model": served,
        "messages": [
            {
                "role": "user",
                "content": long_prompt(
                    int(os.environ.get("YUNSHU_SMOKE_LONG_WORDS", "24000"))
                )
                + "\nSummarize in one word.",
            }
        ],
        "max_tokens": 8,
        "temperature": 0,
        "stream": True,
    }
    progress: list[dict] = []
    saw_data = False
    with httpx.stream(
        "POST",
        f"{BASE}/v1/chat/completions",
        json=body,
        headers={"X-Request-Id": "smoke-long-1"},
        timeout=None,
    ) as r:
        check(
            "long stream headers",
            r.headers.get("x-request-id") == "smoke-long-1"
            and "x-yunshu-queue-position" in r.headers,
        )
        for line in r.iter_lines():
            if line.startswith(": yunshu-progress "):
                progress.append(json.loads(line[len(": yunshu-progress ") :]))
            elif line.startswith("data: "):
                saw_data = True
    pref = [p for p in progress if p.get("phase") == "prefill"]
    check(
        "long prefill emitted progress comments",
        len(progress) >= 1 and saw_data,
        f"{len(progress)} comments, last={progress[-1] if progress else None}",
    )
    if pref:
        check(
            "progress has percent/eta",
            "percent" in pref[-1] and "eta_s" in pref[-1],
            str(pref[-1]),
        )

    # 4. /v1/requests + cancel by X-Request-Id during a long decode.
    cancel_body = {
        "model": served,
        "messages": [
            {"role": "user", "content": "Write a very long story about a lighthouse."}
        ],
        "max_tokens": 4000,
        "temperature": 0.7,
        "stream": True,
    }
    got = {"n": 0, "done": False}

    def consume() -> None:
        with httpx.stream(
            "POST",
            f"{BASE}/v1/chat/completions",
            json=cancel_body,
            headers={"X-Request-Id": "smoke-cancel-1"},
            timeout=None,
        ) as r:
            for line in r.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    got["n"] += 1
        got["done"] = True

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    seen = None
    last_poll = ""
    for _ in range(100):
        time.sleep(0.1)
        rr = httpx.get(f"{BASE}/v1/requests/smoke-cancel-1")
        last_poll = rr.text[:200]
        if rr.status_code == 200 and rr.json().get("phase") == "decode":
            seen = rr.json()
            break
    check(
        "GET /v1/requests/{id} sees the live request",
        seen is not None,
        str(seen or last_poll),
    )
    listing = httpx.get(f"{BASE}/v1/requests").json()
    check(
        "GET /v1/requests lists it",
        any(x["request_id"] == "smoke-cancel-1" for x in listing["data"]),
    )
    status = httpx.get(f"{BASE}/v1/yunshu/status").json()
    check(
        "status: models/memory/requests",
        bool(status["models"])
        and "active_gb" in status["memory"]
        and status["requests"]["active"] >= 1,
        json.dumps({k: status[k] for k in ("state", "uptime_s", "throughput")})[:300],
    )
    d = httpx.delete(f"{BASE}/v1/requests/smoke-cancel-1")
    check("DELETE /v1/requests/{id} cancels", d.status_code == 200, d.text)
    t.join(timeout=30)
    check(
        "cancelled stream ended early",
        got["done"] and got["n"] < 3000,
        f"{got['n']} chunks",
    )
    time.sleep(0.5)
    check(
        "request gone after cancel",
        httpx.get(f"{BASE}/v1/requests/smoke-cancel-1").status_code == 404,
    )

    # 5. Anthropic SDK still works.
    an = anthropic.Anthropic(base_url=BASE, api_key="x", max_retries=0)
    msg = an.messages.create(
        model=served, max_tokens=24, messages=[{"role": "user", "content": "Say hi."}]
    )
    check("anthropic messages", bool(msg.content) and msg.usage.output_tokens > 0)
    with an.messages.stream(
        model=served, max_tokens=24, messages=[{"role": "user", "content": "Say hi."}]
    ) as s:
        list(s)  # thinking models may spend the whole budget on reasoning: count tokens
        final = s.get_final_message()
    check("anthropic stream", final.usage.output_tokens > 0)

    # 6. Ollama SDK still works.
    ol = ollama.Client(host=BASE)
    r1 = ol.chat(
        model=served,
        messages=[{"role": "user", "content": "Say hi."}],
        options={"num_predict": 24},
    )
    check("ollama chat", r1.eval_count > 0 and r1.done)
    parts = list(
        ol.chat(
            model=served,
            messages=[{"role": "user", "content": "Say hi."}],
            options={"num_predict": 24},
            stream=True,
        )
    )
    check("ollama chat stream", len(parts) > 1 and parts[-1].done)

    more_checks(oa, an, ol, served)

    # 7. Warmup.
    w = httpx.post(
        f"{BASE}/v1/yunshu/warmup",
        json={"prompt": "You are a helpful assistant."},
        timeout=120,
    ).json()
    check(
        "warmup",
        w.get("generated") is True and "x_yunshu" in w,
        json.dumps({k: w.get(k) for k in ("model", "load_ms", "warmup_ms")}),
    )

    # 8. Error carries a hint and the request id; SDK exposes the message.
    try:
        oa.chat.completions.create(
            model=served,
            messages=[{"role": "user", "content": "x"}],
            max_tokens=-5,
            extra_headers={"X-Request-Id": "smoke-err-1"},
        )
        check("bad request rejected", False)
    except openai.BadRequestError as e:
        check("error message carries hint", "hint:" in str(e), str(e)[:200])
        body = e.body if isinstance(e.body, dict) else {}
        check(
            "error x_yunshu.request_id",
            (body.get("x_yunshu") or {}).get("request_id") == "smoke-err-1",
            str(body)[:200],
        )


def tee_client(mod, sink: list):
    """An ``mod.Client`` (httpx or httpx2) whose response bodies are also copied to ``sink``."""

    class Tee(mod.SyncByteStream):
        def __init__(self, inner):
            self.inner = inner

        def __iter__(self):
            for chunk in self.inner:
                sink.append(chunk)
                yield chunk

        def close(self):
            self.inner.close()

    class Transport(mod.BaseTransport):
        def __init__(self):
            self.inner = mod.HTTPTransport()

        def handle_request(self, request):
            r = self.inner.handle_request(request)
            return mod.Response(
                r.status_code,
                headers=r.headers,
                stream=Tee(r.stream),
                extensions=r.extensions,
            )

    return mod.Client(transport=Transport(), timeout=None)


def anthropic_http_module(anthropic):
    try:
        import httpx2

        if hasattr(anthropic._base_client, "httpx2"):
            return httpx2
    except ImportError:
        pass
    return httpx


def more_checks(oa, an, ol, served: str) -> None:
    """The 0.1.2 API gaps: Ollama ids/keep_alive/durations, x_yunshu on Messages and
    Responses, progress comments with the Anthropic SDK, Responses fields."""
    import anthropic
    import openai

    # ── Ollama: X-Request-Id, keep_alive, per-response stats in Ollama's own fields ──
    r = httpx.post(
        f"{BASE}/api/chat",
        json={
            "model": served,
            "stream": False,
            "keep_alive": "5m",
            "messages": [{"role": "user", "content": "Say hi in three words."}],
            "options": {"num_predict": 24, "temperature": 0},
        },
        headers={"X-Request-Id": "smoke-ollama-1"},
        timeout=120,
    )
    j = r.json()
    check(
        "ollama echoes X-Request-Id and accepts keep_alive",
        r.status_code == 200 and r.headers.get("x-request-id") == "smoke-ollama-1",
        f"{r.status_code} {r.headers.get('x-request-id')}",
    )
    keys = (
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    )
    check(
        "ollama durations (ns) and counts from engine stats",
        all(isinstance(j.get(k), int) for k in keys)
        and j["eval_count"] > 0
        and j["eval_duration"] > 0
        and j["prompt_eval_count"] > 0
        and j["prompt_eval_duration"] > 0
        and j["total_duration"] >= j["eval_duration"],
        json.dumps({k: j.get(k) for k in keys}),
    )
    sdk = ol.chat(
        model=served,
        messages=[{"role": "user", "content": "Say hi."}],
        options={"num_predict": 16},
        keep_alive=0,
    )
    check(
        "ollama SDK: eval_duration / prompt_eval_duration / total_duration",
        bool(sdk.eval_duration and sdk.prompt_eval_duration and sdk.total_duration),
        f"{sdk.eval_duration} {sdk.prompt_eval_duration} {sdk.total_duration}",
    )
    got = {"n": 0, "done": False}

    def consume() -> None:
        with httpx.stream(
            "POST",
            f"{BASE}/api/chat",
            json={
                "model": served,
                "messages": [{"role": "user", "content": "Write a long story."}],
                "options": {"num_predict": 4000},
            },
            headers={"X-Request-Id": "smoke-ollama-cancel"},
            timeout=None,
        ) as resp:
            for line in resp.iter_lines():
                if line:
                    got["n"] += 1
        got["done"] = True

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    seen = None
    for _ in range(100):
        time.sleep(0.1)
        rr = httpx.get(f"{BASE}/v1/requests/smoke-ollama-cancel")
        if rr.status_code == 200:
            seen = rr.json()
            break
    check("ollama request id reaches /v1/requests/{id}", seen is not None, str(seen))
    httpx.delete(f"{BASE}/v1/requests/smoke-ollama-cancel")
    t.join(timeout=30)
    check("cancel by id stops an Ollama stream", got["done"] and got["n"] < 3000)

    # ── Anthropic: x_yunshu in usage (non-stream + message_delta) ──
    msg = an.messages.create(
        model=served, max_tokens=24, messages=[{"role": "user", "content": "Say hi."}]
    )
    axy = (msg.usage.model_extra or {}).get("x_yunshu") or {}
    check(
        "anthropic message usage.x_yunshu",
        axy.get("ttft_ms") is not None and axy.get("decode_tps") is not None,
        json.dumps(axy)[:240],
    )
    delta_xy = None
    with an.messages.stream(
        model=served, max_tokens=24, messages=[{"role": "user", "content": "Say hi."}]
    ) as st:
        for ev in st:
            if ev.type == "message_delta":
                delta_xy = (ev.usage.model_extra or {}).get("x_yunshu")
        final = st.get_final_message()
    check(
        "anthropic stream: message_delta usage.x_yunshu, SDK final message intact",
        bool(delta_xy and delta_xy.get("ttft_ms")) and final.usage.output_tokens > 0,
        json.dumps(delta_xy)[:240],
    )

    # ── Anthropic SDK on a long prompt: progress comments arrive, the SDK ignores them ──
    words = int(os.environ.get("YUNSHU_SMOKE_LONG_WORDS", "24000"))
    sink: list[bytes] = []
    mod = anthropic_http_module(anthropic)
    tee = anthropic.Anthropic(
        base_url=BASE,
        api_key="x",
        max_retries=0,
        http_client=tee_client(mod, sink),
    )
    with tee.messages.stream(
        model=served,
        max_tokens=8,
        messages=[{"role": "user", "content": long_prompt(words) + "\nOne word."}],
        extra_headers={"X-Request-Id": "smoke-anthropic-long"},
    ) as st:
        list(st)
        fin = st.get_final_message()
    raw = b"".join(sink).decode("utf-8", "replace")
    prog = [
        json.loads(ln[len(": yunshu-progress ") :])
        for ln in raw.splitlines()
        if ln.startswith(": yunshu-progress ")
    ]
    check(
        "anthropic SDK long prompt: progress comments, message parsed",
        len(prog) >= 1 and fin.usage.output_tokens > 0,
        f"{len(prog)} comments, last={prog[-1] if prog else None}",
    )
    pref = [p for p in prog if p.get("phase") == "prefill"]
    if pref:
        check(
            "anthropic progress has percent/eta",
            "percent" in pref[-1] and "eta_s" in pref[-1],
            str(pref[-1]),
        )

    # ── Responses: x_yunshu in usage, echoed config, max_tool_calls, conversation ──
    resp = oa.responses.create(
        model=served,
        input="Say hi in three words.",
        instructions="Be brief.",
        max_output_tokens=32,
        temperature=0,
        truncation="auto",
        max_tool_calls=1,
        extra_headers={"X-Request-Id": "smoke-resp-1"},
    )
    rxy = (resp.usage.model_extra or {}).get("x_yunshu") or {}
    check(
        "responses usage.x_yunshu",
        rxy.get("ttft_ms") is not None and rxy.get("decode_tps") is not None,
        json.dumps(rxy)[:240],
    )
    check(
        "responses echoes instructions / truncation / max_output_tokens / max_tool_calls",
        resp.instructions == "Be brief."
        and resp.truncation == "auto"
        and resp.max_output_tokens == 32
        and resp.max_tool_calls == 1,
        f"{resp.instructions!r} {resp.truncation} {resp.max_output_tokens} {resp.max_tool_calls}",
    )
    events = list(
        oa.responses.create(
            model=served, input="Say hi.", max_output_tokens=32, stream=True
        )
    )
    created = next(e for e in events if e.type == "response.created")
    last = events[-1]
    check(
        "responses stream: created carries config, terminal usage.x_yunshu",
        created.response.truncation == "disabled"
        and last.type in ("response.completed", "response.incomplete")
        and bool(
            ((last.response.usage.model_extra or {}).get("x_yunshu") or {}).get(
                "ttft_ms"
            )
        ),
        f"{created.response.truncation} {last.type}",
    )
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]
    tr = oa.responses.create(
        model=served,
        input="Call get_weather for Paris and also for Tokyo.",
        tools=tools,
        max_tool_calls=1,
        max_output_tokens=256,
        temperature=0,
    )
    calls = [o for o in tr.output if o.type == "function_call"]
    check("responses max_tool_calls=1 caps the calls", len(calls) <= 1, f"{len(calls)}")
    try:
        oa.responses.create(model=served, input="x", conversation="conv_1")
        check("responses conversation rejected", False)
    except openai.BadRequestError as e:
        check(
            "responses conversation rejected with a pointer",
            "previous_response_id" in str(e),
            str(e)[:160],
        )
    trunc_words = int(os.environ.get("YUNSHU_SMOKE_TRUNC_WORDS", "0"))
    if trunc_words:
        items = []
        for i in range(12):
            role = "user" if i % 2 == 0 else "assistant"
            items.append(
                {"role": role, "content": long_prompt(trunc_words) + f" turn{i}"}
            )
        items.append({"role": "user", "content": "Say hi."})
        try:
            oa.responses.create(model=served, input=items, max_output_tokens=8)
            check("truncation disabled overflows", False)
        except openai.BadRequestError as e:
            check(
                "truncation disabled -> 400 context overflow",
                "too long" in str(e).lower(),
            )
        ok = oa.responses.create(
            model=served, input=items, max_output_tokens=8, truncation="auto"
        )
        check(
            "truncation auto drops the oldest input items",
            ok.status in ("completed", "incomplete") and ok.usage.input_tokens > 0,
            f"{ok.status} input_tokens={ok.usage.input_tokens}",
        )


if __name__ == "__main__":
    sys.exit(main())

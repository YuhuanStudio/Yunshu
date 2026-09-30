"""Real-server smoke for the non-SSE transports.

    python scripts/dev/transport_smoke.py --model PATH [--mode tcp|uds|realtime-audio] [--port 18990]

Starts `yunshu serve` (killed with SIGKILL on exit), waits for readiness, then:
  tcp:            /v1/stream (chat, messages, responses, multiplex, cancel, stop, update),
                  Responses WebSocket mode through the official openai SDK
                  (client.responses.connect), Realtime GA + beta conformance.
  uds:            serve --uds; curl --unix-socket, openai SDK over an httpx uds transport,
                  YunshuStream over the socket.
  realtime-audio: as tcp but the Realtime conformance adds the audio path.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, "python")
sys.path.insert(0, os.path.dirname(__file__))

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def start_server(model: str, args: list[str]) -> subprocess.Popen:
    env = dict(os.environ, PYTHONPATH="python")
    return subprocess.Popen(
        [sys.executable, "-m", "yunshu_cli", "serve", "--model", model, *args],
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        start_new_session=True,
    )


def kill9(p: subprocess.Popen) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(p.pid, signal.SIGKILL)
    p.wait()


def wait_ready(check_fn, proc, timeout=420):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("server exited")
        try:
            if check_fn():
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    raise TimeoutError("server not ready")


async def stream_checks(ws_url: str, http_url: str, model: str, uds: str | None = None):
    from yunshu_client import YunshuStream

    async with YunshuStream(ws_url, uds=uds) as conn:
        check(
            "stream: session.created limits",
            conn.session["limits"]["max_inflight"] >= 1,
        )

        async def run(api, body, **kw):
            evs = []
            async for m in conn.request(api, body, with_done=True, **kw):
                evs.append(m)
            return evs

        chat = {
            "model": model,
            "max_tokens": 400,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": "Say hello in five words."}],
        }
        evs = await run("chat.completions", chat, id="chat1")
        deltas = "".join(
            (e["data"]["choices"] or [{}])[0].get("delta", {}).get("content") or ""
            for e in evs
            if e["type"] == "event" and e["data"].get("choices")
        )
        done = evs[-1]
        check(
            "stream: chat.completions text + done",
            bool(deltas) and done["type"] == "done" and done["reason"] == "completed",
            repr(deltas[:50]),
        )
        check(
            "stream: stats ttft/duration",
            done["stats"]["ttft_ms"] is not None,
            str(done["stats"]),
        )
        check(
            "stream: usage chunk delivered",
            any(e["type"] == "event" and e["data"].get("usage") for e in evs),
        )

        evs = await run(
            "responses",
            {"model": model, "input": "Say hi.", "max_output_tokens": 400},
            id="resp1",
        )
        types = [e["data"].get("type") for e in evs if e["type"] == "event"]
        check(
            "stream: responses events",
            "response.created" in types
            and ("response.completed" in types or "response.incomplete" in types),
            json.dumps(
                [
                    e["data"].get("response", {}).get("incomplete_details")
                    for e in evs
                    if e["type"] == "event"
                    and e["data"].get("type") == "response.incomplete"
                ]
            ),
        )

        evs = await run(
            "messages",
            {
                "model": model,
                "max_tokens": 400,
                "messages": [{"role": "user", "content": "Say hi."}],
            },
            id="msg1",
        )
        types = [e["data"].get("type") for e in evs if e["type"] == "event"]
        check(
            "stream: messages events",
            "message_start" in types and "message_stop" in types,
            str(sorted(set(types)))[:200],
        )

        # multiplex: two concurrent chats on one socket
        long = {
            "model": model,
            "max_tokens": 300,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "user", "content": "Write a long story about a dragon."}
            ],
        }
        a, b = await asyncio.gather(
            run("chat.completions", chat, id="ma"),
            run("chat.completions", chat, id="mb"),
        )
        check(
            "stream: 2 concurrent requests complete",
            a[-1]["reason"] == "completed" and b[-1]["reason"] == "completed",
        )

        # cancel mid-generation, then a fresh request must be quick (GPU freed)
        got = 0
        cancel_done = None
        async for m in conn.chat(long, id="cx", with_done=True):
            if m["type"] == "event":
                got += 1
                if got == 5:
                    t_cancel = time.perf_counter()
                    await conn.cancel("cx")
            elif m["type"] == "done":
                cancel_done = m
        check(
            "stream: cancel -> done(cancelled)",
            cancel_done and cancel_done["reason"] == "cancelled",
            str(cancel_done and cancel_done["stats"]),
        )
        t0 = time.perf_counter()
        evs = await run("chat.completions", {**chat, "max_tokens": 4}, id="after")
        check(
            "stream: request after cancel completes promptly",
            evs[-1]["reason"] == "completed",
            f"{time.perf_counter() - t0:.2f}s (cancel->done {time.perf_counter() - t_cancel:.2f}s incl. this)",
        )

        # update max_tokens mid-flight
        n = 0
        upd = None
        async for m in conn.chat(long, id="up", with_done=True):
            if m["type"] == "event":
                n += 1
                if n == 3:
                    await conn.set_max_tokens("up", 10)
            elif m["type"] == "done":
                upd = m
        check(
            "stream: update max_tokens stops early",
            upd and upd["reason"] == "max_tokens" and upd["stats"]["deltas"] <= 40,
            str(upd and upd["stats"]),
        )

        # error path
        try:
            async for _ in conn.chat({"model": model, "messages": "nope"}, id="bad"):
                pass
            check("stream: invalid body -> error", False)
        except Exception as exc:  # noqa: BLE001
            check(
                "stream: invalid body -> error",
                getattr(exc, "status", 0) in (400, 422),
                str(exc)[:100],
            )


async def responses_ws_checks(base: str, model: str):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=base + "/v1",
        api_key="x",
        websocket_base_url=base.replace("http", "ws", 1) + "/v1",
    )
    async with client.responses.connect() as conn:
        await conn.response.create(
            model=model, input="Say hello in three words.", max_output_tokens=400
        )
        types = []
        text = ""
        async for ev in conn:
            types.append(ev.type)
            last = ev
            if ev.type in (
                "response.output_text.delta",
                "response.reasoning_text.delta",
                "response.reasoning_summary_text.delta",
            ):
                text += ev.delta
            if ev.type in (
                "response.completed",
                "response.failed",
                "response.incomplete",
                "error",
            ):
                break
        check(
            "responses-ws: SDK connect + response.create -> completed",
            "response.created" in types
            and types[-1] in ("response.completed", "response.incomplete"),
            f"{types[-1]} {text[:40]!r} {getattr(last, 'response', None) and last.response.incomplete_details}",
        )
        check("responses-ws: text deltas", bool(text.strip()))
        # continuation on the same socket with previous_response_id
        # (server-stored response)


async def realtime_checks(base: str, model: str, audio: bool):
    import realtime_conformance as rc

    class A:
        url = base
        api_key = "x"
        uds = None

    A.model = model
    A.audio = audio
    A.audio_out = False
    A.content = True
    rc.RESULTS.clear()
    await rc.main(A)
    for name, ok, _detail in rc.RESULTS:
        if not ok:
            FAILS.append("realtime: " + name)


def http_ready(url):
    return lambda: (
        urllib.request.urlopen(url + "/health/ready", timeout=3).status == 200
    )


def mode_tcp(a):
    base = f"http://127.0.0.1:{a.port}"
    p = start_server(a.model, ["--port", str(a.port)])
    try:
        wait_ready(http_ready(base), p)
        model = json.load(urllib.request.urlopen(base + "/v1/models"))["data"][0]["id"]
        print("model id:", model, flush=True)
        asyncio.run(stream_checks(f"ws://127.0.0.1:{a.port}/v1/stream", base, model))
        asyncio.run(responses_ws_checks(base, model))
        asyncio.run(realtime_checks(base, model, audio=a.mode == "realtime-audio"))
    finally:
        kill9(p)


def mode_uds(a):
    import subprocess as sp

    sock = tempfile.mkdtemp(prefix="ys", dir="/tmp") + "/y.sock"
    p = start_server(a.model, ["--uds", sock])
    try:

        def ready():
            out = sp.run(
                [
                    "curl",
                    "-s",
                    "--unix-socket",
                    sock,
                    "-o",
                    "/dev/null",
                    "-w",
                    "%{http_code}",
                    "http://localhost/health/ready",
                ],
                capture_output=True,
                text=True,
            )
            return out.stdout == "200"

        wait_ready(ready, p)
        out = sp.run(
            ["curl", "-s", "--unix-socket", sock, "http://localhost/v1/models"],
            capture_output=True,
            text=True,
        )
        model = json.loads(out.stdout)["data"][0]["id"]
        check("uds: curl --unix-socket /v1/models", bool(model), model)
        body = json.dumps(
            {
                "model": model,
                "max_tokens": 300,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": "Say hi."}],
            }
        )
        out = sp.run(
            [
                "curl",
                "-s",
                "--unix-socket",
                sock,
                "-H",
                "content-type: application/json",
                "-d",
                body,
                "http://localhost/v1/chat/completions",
            ],
            capture_output=True,
            text=True,
        )
        check(
            "uds: curl chat completion",
            bool(json.loads(out.stdout)["choices"][0]["message"]["content"]),
        )
        out = sp.run(
            [
                "curl",
                "-sN",
                "--unix-socket",
                sock,
                "-H",
                "content-type: application/json",
                "-d",
                body.replace("}]}", '}],"stream":true}'),
                "http://localhost/v1/chat/completions",
            ],
            capture_output=True,
            text=True,
        )
        check("uds: curl SSE stream", "data: [DONE]" in out.stdout)

        import httpx
        from openai import OpenAI

        oa = OpenAI(
            base_url="http://yunshu/v1",
            api_key="x",
            http_client=httpx.Client(transport=httpx.HTTPTransport(uds=sock)),
        )
        r = oa.chat.completions.create(
            model=model,
            max_tokens=300,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            messages=[{"role": "user", "content": "Say hi."}],
        )
        check(
            "uds: openai SDK over httpx uds transport",
            bool(r.choices[0].message.content),
        )
        chunks = list(
            oa.chat.completions.create(
                model=model,
                max_tokens=300,
                stream=True,
                messages=[{"role": "user", "content": "Say hi."}],
            )
        )
        check("uds: openai SDK streaming over uds", len(chunks) > 2)
        asyncio.run(
            stream_checks("ws://yunshu/v1/stream", "http://yunshu", model, uds=sock)
        )
    finally:
        kill9(p)
        with contextlib.suppress(OSError):
            os.unlink(sock)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="tcp", choices=["tcp", "uds", "realtime-audio"])
    ap.add_argument("--port", type=int, default=18990)
    a = ap.parse_args()
    (mode_uds if a.mode == "uds" else mode_tcp)(a)
    print(f"\nFAILED: {FAILS}" if FAILS else "\nALL PASS", flush=True)
    sys.exit(1 if FAILS else 0)

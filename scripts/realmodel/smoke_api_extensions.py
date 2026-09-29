"""Real-server smoke test of the Yunshu API extensions with the official SDKs.

Starts `uvicorn yunshu_gateway.main:app` on a private port with a small model, then checks that
the OpenAI, Anthropic and Ollama SDKs keep working and that request ids, prefill progress
comments, x_yunshu stats, queue headers, /v1/requests, cancel-by-id, warmup and error hints
show up. Run it through the GPU queue with the SDK venv:

    scripts/dev/gpuq run --priority 1 --timeout 5 --stall 2 --label api-ext-smoke -- \
        /Volumes/P5Plus/yunshu-test-cache/api-ext/sdkvenv/bin/python \
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
    "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python",
)
MODEL = os.environ.get(
    "YUNSHU_SMOKE_MODEL", "/Volumes/P5Plus/models/Qwen3.5-0.8B-MLX-bf16"
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
    for _ in range(100):
        time.sleep(0.1)
        rr = httpx.get(f"{BASE}/v1/requests/smoke-cancel-1")
        if rr.status_code == 200 and rr.json().get("phase") == "decode":
            seen = rr.json()
            break
    check("GET /v1/requests/{id} sees the live request", seen is not None, str(seen))
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


if __name__ == "__main__":
    sys.exit(main())

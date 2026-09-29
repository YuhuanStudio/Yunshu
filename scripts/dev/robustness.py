"""Black-box robustness probe for a Yunshu server (one GPU-queue job).

    gpuq run --priority 1 -- python scripts/dev/robustness.py MODEL_DIR [--only NAME,...]

Starts its own servers on ports 18990-18999, runs each scenario and prints a
JSON table of pass/fail plus the measurements (memory, timings).
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import json
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from pathlib import Path

import httpx

PORT_MAIN, PORT_FAIL, PORT_SHUT = 18991, 18992, 18993
RESULTS: list[dict] = []
LONG_REPS = 6000  # prompt size for the prefill-disconnect probe (~10 tokens each)


def record(name: str, ok: bool, **detail) -> None:
    RESULTS.append({"scenario": name, "ok": bool(ok), **detail})
    print(
        f"[{'PASS' if ok else 'FAIL'}] {name} {json.dumps(detail, default=str)}",
        flush=True,
    )


class Server:
    def __init__(self, model: str, port: int, extra_env: dict | None = None):
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        env = dict(os.environ, YUNSHU_AUTH_DISABLED="true", **(extra_env or {}))
        self.log = tempfile.NamedTemporaryFile(  # noqa: SIM115
            "w+", suffix=".log", delete=False
        )
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "yunshu_cli",
                "serve",
                "-m",
                model,
                "--port",
                str(port),
            ],
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )

    def wait_ready(self, timeout=240.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                return False
            try:
                if httpx.get(self.base + "/health/ready", timeout=2).status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        return False

    def wait_up(self, timeout=60.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                httpx.get(self.base + "/health/live", timeout=2)
                return True
            except httpx.HTTPError:
                time.sleep(0.3)
        return False

    def metric(self, kind="active") -> int:
        text = httpx.get(self.base + "/metrics", timeout=5).text
        m = re.search(rf'yunshu_gpu_memory_bytes\{{type="{kind}"\}} (\d+)', text)
        return int(m.group(1)) if m else -1

    def rss(self) -> int:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(self.proc.pid)],
            capture_output=True,
            text=True,
        )
        return int(out.stdout.strip() or 0) * 1024

    def stop(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def tail(self, n=25) -> str:
        self.log.flush()
        return "\n".join(Path(self.log.name).read_text().splitlines()[-n:])


def chat(base, content="Say hi.", *, max_tokens=16, stream=False, timeout=120, **kw):
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        **kw,
    }
    if stream:
        body["stream"] = True
    return httpx.post(base + "/v1/chat/completions", json=body, timeout=timeout)


def png_data_url() -> str:
    w = h = 64
    raw = b"".join(b"\x00" + bytes([200, 30, 30]) * w for _ in range(h))

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


# ── scenarios ──────────────────────────────────────────────────────────
def load_failures(model: str):
    tmp = Path(tempfile.mkdtemp())
    trunc = tmp / "trunc"
    trunc.mkdir()
    for f in Path(model).iterdir():
        if f.suffix == ".safetensors":
            data = f.read_bytes()[: 1 << 20]
            (trunc / f.name).write_bytes(data)
        else:
            (trunc / f.name).symlink_to(f)
    cases = {"bad_path": str(tmp / "nope"), "truncated_weights": str(trunc)}
    for name, path in cases.items():
        s = Server(path, PORT_FAIL)
        try:
            up = s.wait_up(90)
            time.sleep(2)
            if not up:
                record(
                    f"load_failure/{name}",
                    False,
                    why="server did not stay up",
                    log=s.tail(),
                )
                continue
            r = httpx.get(s.base + "/health/ready", timeout=5)
            body = r.json()
            c = chat(s.base)
            live = httpx.get(s.base + "/health/live", timeout=5).status_code
            ok = (
                r.status_code == 503
                and bool(body.get("reason"))
                and live == 200
                and 400 <= c.status_code < 600
                and c.status_code != 500
            )
            record(
                f"load_failure/{name}",
                ok,
                ready=r.status_code,
                reason=body.get("reason"),
                chat=c.status_code,
                chat_body=c.text[:200],
            )
        finally:
            s.stop()
    shutil.rmtree(tmp, ignore_errors=True)


def disconnect(s: Server):
    # warm
    chat(s.base)
    base_active = s.metric("active")
    # mid-stream disconnect
    with httpx.stream(
        "POST",
        s.base + "/v1/chat/completions",
        json={
            "model": "m",
            "stream": True,
            "max_tokens": 2000,
            "messages": [
                {"role": "user", "content": "Write a very long story about a dragon."}
            ],
        },
        timeout=60,
    ) as r:
        n = 0
        for line in r.iter_lines():
            if line.startswith("data:"):
                n += 1
            if n >= 5:
                break
    t0 = time.time()
    r2 = chat(s.base, "Say OK.", max_tokens=8)
    lat = time.time() - t0
    time.sleep(1)
    record(
        "disconnect/mid_stream",
        r2.status_code == 200 and lat < 5,
        next_latency_s=round(lat, 2),
        active_delta=s.metric("active") - base_active,
    )

    # Disconnect during prefill. A unique long prompt costs `cold` seconds to
    # prefill; after dropping the client at 0.4 s the next small request must
    # not wait for the rest of it.
    def long_prompt(tag):
        return f"[{tag}] " + "The quick brown fox jumps over the lazy dog. " * LONG_REPS

    def body(text, **k):
        return {
            "model": "m",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": text}],
            **k,
        }

    t0 = time.time()
    httpx.post(
        s.base + "/v1/chat/completions", json=body(long_prompt("cold")), timeout=180
    )
    cold = time.time() - t0
    for mode in ("nonstream", "stream"):
        text = long_prompt(mode)
        try:
            if mode == "nonstream":
                httpx.post(
                    s.base + "/v1/chat/completions", json=body(text), timeout=0.4
                )
            else:
                with httpx.stream(
                    "POST",
                    s.base + "/v1/chat/completions",
                    json=body(text, stream=True),
                    timeout=60,
                ) as r:
                    time.sleep(0.4)
        except httpx.HTTPError:
            pass
        t1 = time.time()
        r3 = chat(s.base, "Say OK.", max_tokens=8)
        wait = time.time() - t1
        record(
            f"disconnect/prefill_{mode}",
            r3.status_code == 200 and wait < max(2.0, cold * 0.5),
            next_request_wait_s=round(wait, 2),
            cold_prefill_s=round(cold, 2),
            active_delta=s.metric("active") - base_active,
        )


def limits(s: Server):
    r = chat(s.base, max_tokens=10**9)
    record(
        "limits/max_tokens_huge",
        400 <= r.status_code < 500,
        status=r.status_code,
        body=r.text[:200],
    )
    huge = "hello world " * 300000  # far beyond the 262K window
    r = chat(s.base, huge, max_tokens=8, timeout=120)
    record(
        "limits/context_overflow",
        400 <= r.status_code < 500,
        status=r.status_code,
        body=r.text[:200],
    )
    r = httpx.post(
        s.base + "/v1/chat/completions", json={"model": "m", "messages": []}, timeout=10
    )
    record("limits/empty_messages", 400 <= r.status_code < 500, status=r.status_code)
    r = chat(s.base, max_tokens=-5)
    record(
        "limits/negative_max_tokens", 400 <= r.status_code < 500, status=r.status_code
    )
    ok = chat(s.base).status_code == 200
    record("limits/still_serving", ok)


def concurrent_mix(s: Server):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "p",
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
                "additionalProperties": False,
            },
        },
    }
    img = [
        {"type": "text", "text": "What color is this?"},
        {"type": "image_url", "image_url": {"url": png_data_url()}},
    ]

    def job(kind):
        try:
            if kind == "tools":
                r = chat(
                    s.base,
                    "Weather in Paris?",
                    max_tokens=200,
                    tools=tools,
                    temperature=0,
                )
                return (
                    kind,
                    r.status_code == 200
                    and r.json()["choices"][0]["message"].get("tool_calls") is not None,
                    r.text[:120],
                )
            if kind == "schema":
                r = chat(
                    s.base,
                    "Return a JSON object for a person named Alice who is 30 years old.",
                    max_tokens=200,
                    response_format=schema,
                    temperature=0,
                    chat_template_kwargs={"enable_thinking": False},
                )
                if r.status_code != 200:
                    return kind, False, r.text[:300]
                obj = json.loads(r.json()["choices"][0]["message"]["content"])
                return kind, set(obj) == {"name", "age"}, ""
            if kind == "image":
                r = chat(s.base, img, max_tokens=24, temperature=0)
                return kind, r.status_code == 200, r.text[:120]
            if kind == "stop":
                r = chat(
                    s.base,
                    "Count: 1 2 3 4 5 6 7 8 9 10",
                    max_tokens=60,
                    stop=["5"],
                    temperature=0,
                )
                return (
                    kind,
                    r.status_code == 200
                    and "5" not in r.json()["choices"][0]["message"]["content"],
                    "",
                )
            r = chat(s.base, "Say hi", max_tokens=16, stream=True)
            return kind, r.status_code == 200 and "[DONE]" in r.text, ""
        except Exception as e:  # noqa: BLE001
            return kind, False, repr(e)[:200]

    kinds = ["tools", "schema", "image", "stop", "stream"] * 3
    with cf.ThreadPoolExecutor(15) as ex:
        out = list(ex.map(job, kinds))
    bad = [(k, d) for k, ok, d in out if not ok]
    record("concurrent/mixed_features", not bad, total=len(out), failures=bad[:5])
    record("concurrent/still_serving", chat(s.base).status_code == 200)


def memory(s: Server, n=500):
    chat(s.base)
    samples = []
    for i in range(1, n + 1):
        r = chat(s.base, f"Give one fact about number {i}.", max_tokens=12)
        assert r.status_code == 200, r.text
        if i % 100 == 0 or i == 1:
            samples.append(
                {
                    "i": i,
                    "rss_mb": s.rss() // 2**20,
                    "active_mb": s.metric("active") // 2**20,
                    "cache_mb": s.metric("cache") // 2**20,
                }
            )
    print(samples)
    growth_rss = samples[-1]["rss_mb"] - samples[1]["rss_mb"]
    growth_act = samples[-1]["active_mb"] - samples[1]["active_mb"]
    record(
        "memory/500_sequential",
        growth_rss < 300 and growth_act < 200,
        samples=samples,
        rss_growth_mb=growth_rss,
        active_growth_mb=growth_act,
    )


def shutdown(model: str, sig, name):
    s = Server(model, PORT_SHUT)
    if not s.wait_ready():
        record(f"shutdown/{name}", False, why="not ready", log=s.tail())
        return
    results: list = []

    def stream_job():
        try:
            with httpx.stream(
                "POST",
                s.base + "/v1/chat/completions",
                json={
                    "model": "m",
                    "stream": True,
                    "max_tokens": 3000,
                    "messages": [{"role": "user", "content": "Write a long essay."}],
                },
                timeout=60,
            ) as r:
                n = 0
                for _ in r.iter_lines():
                    n += 1
            results.append(("ended", n))
        except Exception as e:  # noqa: BLE001
            results.append(("error", type(e).__name__))

    ths = [threading.Thread(target=stream_job) for _ in range(3)]
    for t in ths:
        t.start()
    time.sleep(2)
    t0 = time.time()
    s.proc.send_signal(sig)
    try:
        code = s.proc.wait(80)
    except subprocess.TimeoutExpired:
        code = None
        s.proc.kill()
    took = time.time() - t0
    for t in ths:
        t.join(20)
    hung = any(t.is_alive() for t in ths)
    record(
        f"shutdown/{name}",
        code is not None and not hung,
        exit=code,
        secs=round(took, 1),
        clients=results,
        log_tail=s.tail(6) if code is None else "",
    )


def schema_probe(s: Server):
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "p",
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
                "additionalProperties": False,
            },
        },
    }

    def one(_):
        r = chat(
            s.base,
            "Return a JSON object for a person named Alice who is 30 years old.",
            max_tokens=200,
            response_format=schema,
            temperature=0,
            chat_template_kwargs={"enable_thinking": False},
        )
        return r.status_code, r.text[:400]

    alone = [one(i) for i in range(3)]
    with cf.ThreadPoolExecutor(6) as ex:
        conc = list(ex.map(one, range(6)))
    record(
        "schema/alone", all(c == 200 for c, _ in alone), sample=alone[0], log=s.tail(40)
    )
    record(
        "schema/concurrent",
        all(c == 200 for c, _ in conc),
        bad=[t for c, t in conc if c != 200][:2],
    )


SCENARIOS = [
    "load",
    "disconnect",
    "limits",
    "concurrent",
    "memory",
    "shutdown",
    "schema",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--only", default=",".join(SCENARIOS))
    ap.add_argument("--requests", type=int, default=500)
    ap.add_argument("--long-reps", type=int, default=6000)
    a = ap.parse_args()
    global LONG_REPS
    LONG_REPS = a.long_reps
    only = set(a.only.split(","))
    if "load" in only:
        load_failures(a.model)
    if only & {"disconnect", "limits", "concurrent", "memory", "schema"}:
        s = Server(a.model, PORT_MAIN)
        try:
            if not s.wait_ready():
                record("main/start", False, log=s.tail())
            else:
                line = [
                    ln
                    for ln in Path(s.log.name).read_text().splitlines()
                    if "Speculative decoding" in ln or "batch runner" in ln
                ]
                record("main/speculative_path", True, log=line[-2:])
                for name, fn in (
                    ("disconnect", disconnect),
                    ("limits", limits),
                    ("concurrent", concurrent_mix),
                    ("schema", schema_probe),
                ):
                    if name in only:
                        try:
                            fn(s)
                        except Exception as e:  # noqa: BLE001
                            record(
                                f"{name}/exception",
                                False,
                                err=repr(e)[:300],
                                log=s.tail(15),
                            )
                if "memory" in only:
                    try:
                        memory(s, a.requests)
                    except Exception as e:  # noqa: BLE001
                        record("memory/exception", False, err=repr(e)[:300])
        finally:
            s.stop()
    if "shutdown" in only:
        shutdown(a.model, signal.SIGTERM, "sigterm")
        shutdown(a.model, signal.SIGINT, "sigint")
    print("\n=== SUMMARY ===")
    for r in RESULTS:
        print(("PASS" if r["ok"] else "FAIL"), r["scenario"])
    sys.exit(0 if all(r["ok"] for r in RESULTS) else 1)


if __name__ == "__main__":
    main()

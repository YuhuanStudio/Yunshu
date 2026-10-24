"""Real-server check of the spec-sourced defaults (defaultfix): starts `yunshu serve` on MODEL
(isolated HOME, port 18990-18999) and verifies, over raw HTTP,

1. Responses without max_output_tokens: x_yunshu.budget.max_tokens_granted > 32768 and its source
   is the model limit (the Codex shape: reasoning, no max_output_tokens);
2. Chat without max_tokens + ignore_eos streams past 32768 tokens (not cut at 32K);
3. reasoning_effort "none": no reasoning content; Anthropic without `thinking`: no thinking block;
4. Anthropic without max_tokens: 400;
5. a 20 MiB body is not refused with 413, a 34 MiB one is.

    python scripts/research/defaultfix_real.py MODEL_DIR --port 18991 [--long-tokens 33000]  (0 skips the long stream)
Exits nonzero on the first failure.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

TOKEN = "defaultfix-secret"


class Fail(AssertionError):  # noqa: N818
    pass


def expect(cond, msg):
    if not cond:
        raise Fail(msg)


def wait_ready(url, proc, secs=300):
    for _ in range(secs):
        if proc.poll() is not None:
            raise Fail(f"server exited rc={proc.returncode}")
        try:
            urllib.request.urlopen(url + "/health/ready", timeout=2)
            return
        except Exception:  # noqa: BLE001
            time.sleep(1)
    raise Fail("server not ready")


def pad_body(body: dict, mib: float) -> bytes:
    """`body` plus an ignored top-level `padding` string so the JSON is about `mib` MiB."""
    base = len(json.dumps(dict(body, padding="")).encode())
    pad = max(0, int(mib * 1024 * 1024) - base)
    return json.dumps(dict(body, padding="x" * pad)).encode()


def post(url, path, body, raw=None, timeout=600):
    data = raw if raw is not None else json.dumps(body).encode()
    req = urllib.request.Request(
        url + path,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOKEN}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raw_err = e.read()
        try:
            return e.code, json.loads(raw_err)
        except ValueError:
            return e.code, {"raw": raw_err[:200].decode("utf-8", "replace")}


def stream_count(url, body, stop_after, timeout=900):
    """Number of content chunks seen on a chat stream, closing it after `stop_after`."""
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(dict(body, stream=True)).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOKEN}",
        },
    )
    n = 0
    finish = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
                continue
            ch = json.loads(line[6:])
            for c in ch.get("choices", []):
                d = c.get("delta", {})
                if d.get("content") or d.get("reasoning_content"):
                    n += 1
                finish = c.get("finish_reason") or finish
            if n >= stop_after or finish:
                break
    return n, finish


def run_checks(url, results, long_tokens):
    def step(name, fn):
        try:
            results[name] = {"ok": True, "detail": fn()}
        except Exception as exc:  # noqa: BLE001
            results[name] = {"ok": False, "detail": repr(exc)[:400]}
            raise

    def responses_default():
        # Codex shape: reasoning object, no max_output_tokens
        st, j = post(
            url,
            "/v1/responses",
            {"model": "m", "input": "Say hi.", "reasoning": {"effort": "high"}},
        )
        expect(st == 200, f"status {st}: {j}")
        b = ((j.get("usage") or {}).get("x_yunshu") or {}).get("budget") or {}
        expect(b.get("max_tokens_granted", 0) > 32768, f"budget {b}")
        expect(b.get("max_tokens_source") == "model_limit", f"budget {b}")
        return b

    def long_not_cut():
        n, fin = stream_count(
            url,
            {
                "model": "m",
                "messages": [{"role": "user", "content": "Count upward forever."}],
                "ignore_eos": True,
                "enable_thinking": False,
            },
            long_tokens,
        )
        expect(n >= long_tokens, f"stream ended at {n} chunks, finish={fin}")
        return {"chunks": n, "finish": fin}

    def effort_none():
        st, j = post(
            url,
            "/v1/chat/completions",
            {
                "model": "m",
                "messages": [{"role": "user", "content": "What is 17*23?"}],
                "reasoning_effort": "none",
                "max_tokens": 300,
            },
        )
        expect(st == 200, f"status {st}: {j}")
        m = j["choices"][0]["message"]
        expect(not m.get("reasoning_content"), f"reasoning present: {m}")
        expect("<think>" not in (m.get("content") or ""), "think tag in content")
        return (m.get("content") or "")[:60]

    def anthropic_no_thinking():
        st, j = post(
            url,
            "/v1/messages",
            {
                "model": "m",
                "max_tokens": 300,
                "messages": [{"role": "user", "content": "What is 17*23?"}],
            },
        )
        expect(st == 200, f"status {st}: {j}")
        kinds = [c.get("type") for c in j.get("content", [])]
        expect("thinking" not in kinds, f"thinking block present: {kinds}")
        return kinds

    def anthropic_max_tokens_required():
        st, j = post(
            url,
            "/v1/messages",
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        expect(st == 400, f"status {st}: {j}")
        expect(j.get("error", {}).get("type") == "invalid_request_error", str(j))
        return j["error"]["message"][:80]

    def body_sizes():
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 8,
        }
        st, _ = post(url, "/v1/chat/completions", None, raw=pad_body(body, 20))
        expect(st != 413, "20 MiB refused with 413")
        st2, _ = post(url, "/v1/chat/completions", None, raw=pad_body(body, 34))
        expect(st2 == 413, f"34 MiB gave {st2}")
        return {"20MiB": st, "34MiB": st2}

    for name, fn in [
        ("responses_default_cap", responses_default),
        ("effort_none_no_reasoning", effort_none),
        ("anthropic_absent_thinking", anthropic_no_thinking),
        ("anthropic_max_tokens_required", anthropic_max_tokens_required),
        ("body_sizes", body_sizes),
    ] + ([("long_generation_not_cut_at_32k", long_not_cut)] if long_tokens else []):
        step(name, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--port", type=int, default=18991)
    ap.add_argument("--long-tokens", type=int, default=33000)
    ap.add_argument("--home", default="/Volumes/P5Plus/yunshu-build/defaultfix-home")
    a = ap.parse_args()
    expect(18990 <= a.port <= 18999, "port outside 18990-18999")
    os.makedirs(a.home, exist_ok=True)
    tree = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(
        os.environ,
        HOME=a.home,
        PYTHONPATH=os.path.join(tree, "python"),
        YUNSHU_AUTH_TOKEN=TOKEN,
        HF_HUB_OFFLINE="1",
    )
    env.pop("YUNSHU_AUTH_DISABLED", None)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "yunshu_cli",
            "serve",
            "--model",
            a.model,
            "--port",
            str(a.port),
        ],
        cwd=tree,
        env=env,
        stdout=open(os.path.join(a.home, "server.log"), "w"),  # noqa: SIM115
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    url = f"http://127.0.0.1:{a.port}"
    results: dict = {}
    rc = 1
    try:
        wait_ready(url, proc)
        run_checks(url, results, a.long_tokens)
        rc = 0
    except BaseException as exc:  # noqa: BLE001
        results["_error"] = repr(exc)[:400]
    finally:
        with __import__("contextlib").suppress(Exception):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    print(json.dumps(results, indent=1, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    sys.exit(main())

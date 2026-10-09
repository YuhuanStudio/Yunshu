"""One engine arm on one Flash-Next pack: decode tok/s and TTFT by streaming, plus system-delta footprint.

  flashnext80_engines.py --arm tf-nodraft|tf-mtp|yunshu-off|yunshu-mtp --model PACK --out F.jsonl
        [--ctx 1024 8192] [--reps 3] [--tokens 192] [--tf-arg ...] [--env K=V]

Existence-proof harness: TensorFold 0.6.1 (``tensorfold serve``) vs Yunshu on the SAME pack. Fail closed:
``complete`` only when every request streamed the full token count. Run through gpuq only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from memory_ab import code_doc, log_tail, stop_server  # noqa: E402
from process_memory import system_used_bytes  # noqa: E402

TF_BIN = "/Volumes/P5Plus/yunshu-test-envs/tensorfold-0.6.1/bin/tensorfold"
YUNSHU_PY = "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python"
GB = 1e9


def parse_sse_line(line: bytes):
    """-> ('delta', text) | ('usage', dict) | ('done', None) | None for one SSE line."""
    line = line.strip()
    if not line.startswith(b"data:"):
        return None
    payload = line[5:].strip()
    if payload == b"[DONE]":
        return ("done", None)
    try:
        d = json.loads(payload)
    except ValueError:
        return None
    if d.get("x_yunshu"):
        return ("x", (d["x_yunshu"] or {}).get("speculative"))
    if d.get("usage") and not d.get("choices"):
        return ("usage", d["usage"])
    ch = (d.get("choices") or [{}])[0]
    delta = ch.get("delta") or {}
    text = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
    return ("delta", text) if text else None


def stream_request(url, content, max_tokens):
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first = last = None
    deltas = 0
    text = []
    usage = None
    spec = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            ev = parse_sse_line(raw)
            if ev is None:
                continue
            kind, val = ev
            if kind == "delta":
                now = time.perf_counter()
                first = first or now
                last = now
                deltas += 1
                text.append(val)
            elif kind == "usage":
                usage = val
            elif kind == "x":
                spec = val
            else:
                break
    return {
        "ttft_s": None if first is None else first - t0,
        "decode_s": None if first is None else last - first,
        "deltas": deltas,
        "usage": usage,
        "spec": spec,
        "text": "".join(text),
    }


def decode_tok_s(res):
    n = (res.get("usage") or {}).get("completion_tokens") or res["deltas"]
    if res["decode_s"] is None or res["decode_s"] <= 0 or n < 2:
        return None
    return (n - 1) / res["decode_s"]


def server_cmd(arm, model, port, tf_args):
    if arm.startswith("tf-"):
        cmd = [TF_BIN, "serve", model, "--port", str(port), "--no-update-check"]
        cmd += ["--no-drafts"] if arm == "tf-nodraft" else []
        return cmd + list(tf_args)
    return [
        YUNSHU_PY,
        "-m",
        "yunshu_cli",
        "serve",
        "--model",
        model,
        "--port",
        str(port),
    ]


def arm_env(arm, extra):
    env = dict(os.environ, YUNSHU_AUTH_DISABLED="1", HF_HUB_OFFLINE="1")
    if arm.startswith("yunshu"):
        env["YUNSHU_VLM_DRAFT"] = "mtp" if arm == "yunshu-mtp" else "off"
        if arm == "yunshu-mtp":
            env.setdefault("YUNSHU_MTP_BLOCK_SIZE", "4")
    env.update(dict(kv.split("=", 1) for kv in extra))
    return env


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        required=True,
        choices=["tf-nodraft", "tf-mtp", "yunshu-off", "yunshu-mtp"],
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=18995)
    ap.add_argument("--ctx", type=int, nargs="+", default=[1024])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--tf-arg", action="append", default=[])
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--tree", default=os.getcwd())
    ap.add_argument("--apc-gb", default="1")
    ap.add_argument("--out", required=True)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    env = arm_env(a.arm, a.env)
    env["PYTHONPATH"] = os.path.join(a.tree, "python")
    if a.arm.startswith("yunshu"):
        env.setdefault("YUNSHU_VLM_APC_MEMORY_GB", a.apc_gb)
    baseline = system_used_bytes()
    rows = []
    out = open(a.out, "w")  # noqa: SIM115

    def emit(row):
        rows.append(row)
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps(row), flush=True)

    log = open(os.path.splitext(a.out)[0] + f".{a.arm}.server.log", "w")  # noqa: SIM115
    cmd = server_cmd(a.arm, a.model, a.port, a.tf_arg)
    emit({"kind": "launch", "arm": a.arm, "cmd": cmd, "env_overrides": a.env})
    proc = subprocess.Popen(
        cmd,
        cwd=a.tree,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    url = f"http://127.0.0.1:{a.port}"
    peak = [0]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak[0] = max(peak[0], system_used_bytes())
            time.sleep(0.25)

    bad = []
    try:
        for _ in range(1800):
            try:
                urllib.request.urlopen(url + "/v1/models", timeout=2)
                break
            except Exception as exc:  # noqa: BLE001
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"server exited rc={proc.returncode}: " + log_tail(log.name)
                    ) from exc
                time.sleep(1)
        else:
            raise RuntimeError("server not ready")
        threading.Thread(target=sampler, daemon=True).start()
        emit(
            {
                "kind": "ready",
                "steady_delta_gb": round((system_used_bytes() - baseline) / GB, 2),
            }
        )
        stream_request(url, code_doc(1, 300) + "\nSummarize.", 16)  # warmup
        for ctx in a.ctx:
            for rep in range(a.reps):
                doc = code_doc(ctx * 100 + rep, max(ctx - 200, 100))
                res = stream_request(
                    url,
                    doc
                    + "\nList three functions that use 'cache'; explain each at length.",
                    a.tokens,
                )
                n = (res["usage"] or {}).get("completion_tokens") or res["deltas"]
                rate = decode_tok_s(res)
                if rate is None or n < a.tokens:
                    bad.append(f"ctx{ctx} rep{rep}: {n} tokens, rate {rate}")
                emit(
                    {
                        "kind": "request",
                        "arm": a.arm,
                        "ctx": ctx,
                        "rep": rep,
                        "ttft_s": res["ttft_s"],
                        "decode_tok_s": rate,
                        "tokens": n,
                        "text_head": res["text"][:160],
                        "spec": res["spec"],
                        "digest": hashlib.sha256(res["text"].encode()).hexdigest()[:16],
                        "steady_delta_gb": round(
                            (system_used_bytes() - baseline) / GB, 2
                        ),
                        "peak_delta_gb": round((peak[0] - baseline) / GB, 2),
                    }
                )
    finally:
        stop.set()
        stop_server(proc)
        log.close()
    emit(
        {
            "complete": not bad and any(r.get("kind") == "request" for r in rows),
            "problems": bad,
        }
    )
    out.close()
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())

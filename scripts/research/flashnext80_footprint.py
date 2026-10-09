"""Server footprint of a large MoE pack by the system-delta method (host vm_stat, not phys_footprint).

Starts ONE server from a tree, takes the host used-memory baseline before launch, then runs a request
ladder and records after every request: steady and peak system delta, MLX active/cache, the PLE-on-SSD
counters (/debug/weight-residency) and the per-request x_yunshu stats (spec rounds, TTFT, decode).
Exit code is nonzero (and nothing says complete) when a required path did not engage.

    python flashnext80_footprint.py --tree TREE --model PACK --sizes 1024 8192 --out out.jsonl \
        [--env K=V ...] [--require-ple] [--require-spec]
Run it through gpuq only.
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

GB = 1e9


def http_json(url, body=None, timeout=1800):
    req = urllib.request.Request(
        url,
        None if body is None else json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def verdict(rows, require_ple, require_spec):
    """Reasons the run is NOT a success (empty list = success). Fail closed on missing evidence."""
    reasons = []
    reqs = [r for r in rows if r.get("kind") == "request"]
    if not reqs:
        return ["no request rows"]
    if any(not r.get("tokens") for r in reqs):
        reasons.append("a request generated no tokens")
    if require_ple:
        ple = [r.get("ple") or {} for r in rows if r.get("kind") == "request"]
        if not ple or not ple[-1].get("engaged"):
            reasons.append("PLE-on-SSD counters did not move")
    if require_spec and not any(
        ((r.get("x_yunshu") or {}).get("speculative") or {}).get("rounds", 0) > 0
        for r in reqs
    ):
        reasons.append("MTP engaged zero verify rounds")
    return reasons


def summarize(rows, baseline):
    reqs = [r for r in rows if r.get("kind") == "request"]
    return {
        "kind": "summary",
        "baseline_gb": round(baseline / GB, 3),
        "peak_delta_gb": max((r["peak_delta_gb"] for r in reqs), default=None),
        "steady_delta_gb": [r["steady_delta_gb"] for r in reqs],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=18994)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 8192])
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--require-ple", action="store_true")
    ap.add_argument("--require-spec", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    env = dict(
        os.environ,
        PYTHONPATH=os.path.join(a.tree, "python"),
        YUNSHU_AUTH_DISABLED="1",
        YUNSHU_DEBUG_ROUTES="1",
    )
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    baseline = system_used_bytes()
    rows = []
    out = open(a.out, "w")  # noqa: SIM115

    def emit(row):
        rows.append(row)
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps(row), flush=True)

    log = open(os.path.splitext(a.out)[0] + ".server.log", "w")  # noqa: SIM115
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
            time.sleep(0.2)

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
                "steady_delta_gb": round((system_used_bytes() - baseline) / GB, 3),
                "peak_delta_gb": round((peak[0] - baseline) / GB, 3),
            }
        )

        def request(label, content, again=None):
            peak[0] = 0
            msgs = [{"role": "user", "content": content}]
            t = time.time()
            d = http_json(
                url + "/v1/chat/completions",
                {
                    "model": "m",
                    "messages": msgs,
                    "max_tokens": a.max_tokens,
                    "temperature": 0,
                },
            )
            m = d["choices"][0]["message"]
            text = (m.get("reasoning_content") or m.get("reasoning") or "") + (
                m.get("content") or ""
            )
            try:
                ple = http_json(url + "/debug/weight-residency", timeout=30)
            except Exception as exc:  # noqa: BLE001
                ple = {"error": str(exc)}
            emit(
                {
                    "kind": "request",
                    "label": label,
                    "secs": round(time.time() - t, 3),
                    "usage": d.get("usage"),
                    "x_yunshu": d.get("x_yunshu"),
                    "tokens": d.get("usage", {}).get("completion_tokens", 0),
                    "digest": hashlib.sha256(text.encode()).hexdigest()[:16],
                    "steady_delta_gb": round((system_used_bytes() - baseline) / GB, 3),
                    "peak_delta_gb": round((peak[0] - baseline) / GB, 3),
                    "ple": ple,
                }
            )

        for size in a.sizes:
            doc = code_doc(size, max(size - 200, 200))
            q = doc + "\nList three functions that use 'cache'."
            request(f"{size}-cold", q)
            request(
                f"{size}-warm", q
            )  # same prompt: APC hit, digest must equal the cold one
        time.sleep(20)
        emit(
            {
                "kind": "idle20s",
                "steady_delta_gb": round((system_used_bytes() - baseline) / GB, 3),
            }
        )
    finally:
        stop.set()
        stop_server(proc)
        log.close()
    reasons = verdict(rows, a.require_ple, a.require_spec)
    emit(summarize(rows, baseline))
    emit({"complete": not reasons, "reasons": reasons})
    out.close()
    return 0 if not reasons else 1


if __name__ == "__main__":
    sys.exit(main())

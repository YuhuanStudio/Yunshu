"""Replay an agent-like growing conversation against the real server and log Metal-buffer counters.

    python scripts/research/metalbuf_repro.py MODEL --out DIR [--turns 32] [--extra-env K=V,K=V]

Each turn: prompt = previous prompt + previous reply + 0.5-3.5K new tokens of repo source (a 'tool
result'), temperature sampling, APC on.  After every turn the client asks the server-side monitor
(metalbuf_serve.py) for a headroom probe.  Exits nonzero on any request error or when the server dies.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import pathlib
import random
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def chunks_of_source(root: str, seed: int):
    files = sorted(
        glob.glob(os.path.join(root, "python", "**", "*.py"), recursive=True)
    )
    rng = random.Random(seed)
    rng.shuffle(files)
    for f in files:
        try:
            lines = pathlib.Path(f).read_text().splitlines()
        except OSError:
            continue
        for i in range(0, len(lines), 150):
            yield f"File {f}:{i}\n" + "\n".join(lines[i : i + 150])


def take_tokens(gen, n_chars: int) -> str:
    out, size = [], 0
    while size < n_chars:
        c = next(gen)
        out.append(c)
        size += len(c)
    return "\n\n".join(out)[:n_chars]


def chat(url, messages, max_tokens, seed):
    body = json.dumps(
        {
            "model": "m",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.6,
            "top_p": 0.95,
            "seed": seed,
        }
    ).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions", body, {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:500]!r}") from e
    msg = d["choices"][0]["message"]
    return msg.get("content") or "", d["usage"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=18991)
    ap.add_argument("--turns", type=int, default=32)
    ap.add_argument("--home", default="/Volumes/P5Plus/yunshu-build/metalbuf-home")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--extra-env", default="")
    ap.add_argument("--draft", default="")
    a = ap.parse_args()
    assert 18990 <= a.port <= 18999
    os.makedirs(a.out, exist_ok=True)
    tree = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(
        os.environ,
        HOME=a.home,
        PYTHONPATH=os.path.join(tree, "python"),
        YUNSHU_AUTH_DISABLED="true",
        HF_HUB_OFFLINE="1",
        METALBUF_OUT=a.out,
    )
    env.pop("YUNSHU_AUTH_TOKEN", None)
    if a.draft:
        env["YUNSHU_VLM_DRAFT"] = a.draft
    for kv in filter(None, a.extra_env.split(",")):
        k, v = kv.split("=", 1)
        env[k] = v
    os.makedirs(a.home, exist_ok=True)
    log = open(os.path.join(a.out, "server.log"), "w")  # noqa: SIM115
    proc = subprocess.Popen(
        [
            sys.executable,
            os.path.join(tree, "scripts/research/metalbuf_serve.py"),
            "serve",
            "--model",
            a.model,
            "--port",
            str(a.port),
        ],
        cwd=tree,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    url = f"http://127.0.0.1:{a.port}"
    rc = 1
    try:
        for _ in range(900):
            if proc.poll() is not None:
                raise RuntimeError(f"server exited rc={proc.returncode}")
            try:
                urllib.request.urlopen(url + "/health/ready", timeout=2)
                break
            except Exception:  # noqa: BLE001
                time.sleep(1)
        else:
            raise RuntimeError("server not ready")
        rng = random.Random(7)
        gen = chunks_of_source(tree, 1)
        messages = [
            {
                "role": "user",
                "content": "You are reviewing a codebase. Read these files and answer briefly.\n\n"
                + take_tokens(gen, 36000),
            }
        ]
        rows = []
        for turn in range(a.turns):
            t0 = time.time()
            reply, usage = chat(url, messages, a.max_tokens, 1000 + turn)
            dt = time.time() - t0
            rows.append(
                {
                    "turn": turn,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "s": round(dt, 1),
                }
            )
            print(json.dumps(rows[-1]), flush=True)
            if proc.poll() is not None:
                raise RuntimeError("server died")
            if False:
                open(os.path.join(a.out, "probe"), "w").close()
                time.sleep(5)
            messages.append({"role": "assistant", "content": reply or "ok"})
            n_chars = rng.randint(1500, 10500)
            messages.append(
                {
                    "role": "user",
                    "content": "Tool result:\n\n"
                    + take_tokens(gen, n_chars)
                    + "\n\nContinue.",
                }
            )
        with open(os.path.join(a.out, "turns.json"), "w") as f:
            json.dump(rows, f)
        rc = 0
    finally:
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)
    return rc


if __name__ == "__main__":
    sys.exit(main())

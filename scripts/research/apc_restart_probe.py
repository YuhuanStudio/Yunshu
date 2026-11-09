"""Real restart probe for the APC SSD tier: serve, one long prompt, stop (term|kill), serve again,
repeat the prompt, require cached > 0. Run under gpuq. Exits nonzero on any failure."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request


def make_prompt(words: int) -> str:
    return " ".join(f"item{i % 97} alpha{i % 13}" for i in range(words // 2))


def cached_of(resp: dict) -> int:
    d = (resp.get("usage") or {}).get("prompt_tokens_details") or {}
    return int(d.get("cached_tokens") or 0)


def post(port: int, prompt: str, timeout: float = 600) -> dict:
    body = json.dumps(
        {
            "model": "m",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        body,
        {"Content-Type": "application/json"},
    )
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    out["_wall"] = time.monotonic() - t
    return out


def serve(model, port, home, log):
    env = dict(os.environ, HOME=home, YUNSHU_MODEL=model)
    p = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "yunshu_cli",
            "serve",
            "--model",
            model,
            "--port",
            str(port),
        ],
        env=env,
        stdout=open(log, "wb"),  # noqa: SIM115
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(300):
        if p.poll() is not None:
            raise SystemExit(f"server exited rc={p.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2)
            return p
        except Exception:
            time.sleep(1)
    p.kill()
    raise SystemExit("server did not become ready")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--port", type=int, default=18990)
    ap.add_argument("--stop", choices=("term", "kill"), default="term")
    ap.add_argument("--words", type=int, default=6000)
    a = ap.parse_args()
    home = os.path.join(a.work, "home")
    os.makedirs(home, exist_ok=True)
    prompt = make_prompt(a.words)
    p = serve(a.model, a.port, home, os.path.join(a.work, "s1.log"))
    try:
        r1 = post(a.port, prompt)
        print("first", cached_of(r1), r1["usage"], round(r1["_wall"], 2), flush=True)
        time.sleep(3)
    finally:
        sig = signal.SIGTERM if a.stop == "term" else signal.SIGKILL
        os.killpg(p.pid, sig)
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
    p = serve(a.model, a.port, home, os.path.join(a.work, "s2.log"))
    try:
        r2 = post(a.port, prompt)
    finally:
        os.killpg(p.pid, signal.SIGKILL)
    c = cached_of(r2)
    print("restart", c, r2["usage"], round(r2["_wall"], 2), flush=True)
    if c <= 0:
        raise SystemExit("FAIL: restart cached=0")


if __name__ == "__main__":
    main()

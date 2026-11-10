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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompt_len  # noqa: E402

FILLER_SOURCE = " ".join(f"item{i % 97} alpha{i % 13}" for i in range(20000))


def make_prompt(tok, ctx_tokens: int) -> str:
    """The prompt whose served turn-1 length (template included) is exactly ctx_tokens (prompt_len.py)."""
    return prompt_len.build_text(tok, FILLER_SOURCE, ctx_tokens)


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
    ap.add_argument(
        "--ctx-tokens",
        type=int,
        default=None,
        help="served turn-1 prompt tokens (template included; prompt_len.py)",
    )
    ap.add_argument(
        "--words",
        type=int,
        default=None,
        help="DEPRECATED: word count of the old prompt; converted to its served tokens with the tokenizer (logged)",
    )
    ap.add_argument(
        "--tokenizer", default=None, help="tokenizer dir (default: --model)"
    )
    a = ap.parse_args()
    if a.ctx_tokens is not None and a.words is not None:
        ap.error("--words is a deprecated alias of --ctx-tokens; give one")
    tok = prompt_len.load_tokenizer(a.tokenizer or a.model)
    if a.ctx_tokens is not None:
        ctx = a.ctx_tokens
    else:  # the deprecated word count (default 6000) becomes the served length of that old prompt
        words = 6000 if a.words is None else a.words
        legacy = " ".join(f"item{i % 97} alpha{i % 13}" for i in range(words // 2))
        ctx = prompt_len.served_tokens(tok, legacy)
        print(
            f"warning: --words {words} is deprecated; using --ctx-tokens {ctx}",
            flush=True,
        )
    home = os.path.join(a.work, "home")
    os.makedirs(home, exist_ok=True)
    prompt = make_prompt(tok, ctx)
    p = serve(a.model, a.port, home, os.path.join(a.work, "s1.log"))
    try:
        r1 = post(a.port, prompt)
        prompt_len.check_served_length(ctx, r1["usage"])
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
        prompt_len.check_served_length(ctx, r2["usage"])
    finally:
        os.killpg(p.pid, signal.SIGKILL)
    c = cached_of(r2)
    print("restart", c, r2["usage"], round(r2["_wall"], 2), flush=True)
    if c <= 0:
        raise SystemExit("FAIL: restart cached=0")


if __name__ == "__main__":
    main()

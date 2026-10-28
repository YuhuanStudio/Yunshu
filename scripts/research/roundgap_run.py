#!/usr/bin/env python3
"""One served-path round probe: start ONE server (Yunshu or TensorFold native), send long-ask requests, record text, rounds,
accepted drafts and the server's own round profile lines. Run through gpuq; the server is SIGKILLed on exit.

  roundgap_run.py --engine yunshu|yunshu-tf4|tf-native --kind code|prose --ctx 1024 --tokens 512 --reps 2 \
      [--dylib trace.dylib|trace2.dylib] [--env K=V ...] --out F.json

Fails closed: nonzero exit if a stream is incomplete, a request is short, or the engine reported no round count.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = Path("/Volumes/P5Plus/yunshu-build/tfnew/prompts")
OQ4E = "/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
TF4 = "/Volumes/P5Plus/models/TensorFold/Qwen3.8-27B-MLX-4bit"
DRAFT = "/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2"
TF_BIN = "/Volumes/P5Plus/yunshu-test-envs/tensorfold-1.0.7/tensorfold-1.0.7-macos-arm64/bin/tensorfold-native"
YUNSHU_BIN = "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/yunshu"

TF_DONE = re.compile(
    r"prompt=(\d+) cached=(\d+).*?tokens=(\d+).*?finish=(\w+).*?rounds=(\d+) accepted=(\d+)/(\d+)"
    r"(?: ms/round=([\d.]+) forward=([\d.]+) draft=([\d.]+) post=([\d.]+) rows=([\d.]+))?"
)
WARM_PROMPT = (
    "Write a Python function that reads a CSV file and prints the sum of every numeric column. "
    "Include comments and a short usage example."
)
KEEP = re.compile(
    r"GPU window|GPU idle by host|Host time by|GPU per-kernel|DFlash round profile|DFlash host between|"
    r"Tree verify stage|suffix graft engagement|VLM batch runner|DFlash drafter|tree_block|drafts: on"
)


def parse_tf_done(line: str) -> dict | None:
    m = TF_DONE.search(line)
    if not m:
        return None
    pt, cached, tokens, finish, rounds, acc, drafted, msr, fwd, drf, post, rows = (
        m.groups()
    )
    return dict(
        prompt=int(pt),
        cached=int(cached),
        tokens=int(tokens),
        finish=finish,
        rounds=int(rounds),
        accepted=int(acc),
        drafted=int(drafted),
        ms_per_round=float(msr) if msr else None,
        forward_ms=float(fwd) if fwd else None,
        draft_ms=float(drf) if drf else None,
        post_ms=float(post) if post else None,
        rows=float(rows) if rows else None,
    )


def round_summary(tokens: int, rounds: int, decode_s: float) -> dict:
    """Split decode speed into tokens per round and ms per round (first token is prefill's)."""
    if rounds <= 0 or tokens < 2 or decode_s <= 0:
        raise ValueError("no round evidence")
    return {
        "tokens_per_round": (tokens - 1) / rounds,
        "wall_ms_per_round": decode_s * 1000 / rounds,
        "tok_s": (tokens - 1) / decode_s,
    }


def free_port() -> int:
    for p in range(18990, 19000):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
            except OSError:
                continue
        return p
    raise RuntimeError("no free port in 18990-18999")


def send(url: str, body: dict, timeout: float = 900) -> dict:
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer k"},
    )
    t0 = time.perf_counter()
    tf = None
    out: list[str] = []
    usage = finish = xy = None
    done = False
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            p = line[5:].strip()
            if p == b"[DONE]":
                done = True
                break
            d = json.loads(p)
            if d.get("error"):
                raise RuntimeError(f"SSE error {d['error']}")
            usage = d.get("usage") or usage
            xy = d.get("x_yunshu") or xy
            for ch in d.get("choices") or []:
                dl = ch.get("delta") or {}
                c = dl.get("content") or dl.get("reasoning_content")
                if c:
                    out.append(c)
                    tf = tf or time.perf_counter()
                finish = ch.get("finish_reason") or finish
    t1 = time.perf_counter()
    if not done or finish is None or usage is None or tf is None:
        raise RuntimeError("incomplete stream")
    return dict(
        text="".join(out),
        ct=usage["completion_tokens"],
        pt=usage["prompt_tokens"],
        cached=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        ttft_s=tf - t0,
        decode_s=t1 - tf,
        finish=finish,
        xy=xy,
    )


def launch(a, home: Path, log: Path):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("ANTHROPIC_", "OPENAI_", "CLAUDE", "CODEX", "YUNSHU_"))
    }
    env.update(HOME=str(home), HF_HUB_OFFLINE="1", NO_PROXY="127.0.0.1")
    port = free_port()
    if a.engine == "tf-native":
        cmd = [
            TF_BIN, "serve", TF4, "--host", "127.0.0.1", "--port", str(port), "--drafter", DRAFT,
            "--no-update-check", "--no-thinking", "--parallel", "1", "--context", str(a.ctx + 4096),
        ]  # fmt: skip
    else:
        model = TF4 if a.engine == "yunshu-tf4" else a.model or OQ4E
        env.update(
            PYTHONPATH=str(ROOT / "python"),
            YUNSHU_CONSOLE="0",
            YUNSHU_VLM_APC_DISK="0",
        )
        if not a.no_drafter:
            env["YUNSHU_VLM_DRAFT"] = a.drafter or DRAFT
        cmd = [YUNSHU_BIN, "serve", "-m", model, "--port", str(port)]
    if a.dylib:
        env["DYLD_INSERT_LIBRARIES"] = a.dylib
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    with open(log, "ab") as lf:
        proc = subprocess.Popen(
            cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
    return proc, f"http://127.0.0.1:{port}"


def wait_ready(proc, url, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("server exited early")
        try:
            with urllib.request.urlopen(url + "/v1/models", timeout=3) as r:
                return json.load(r)["data"][0]["id"]
        except Exception:
            time.sleep(2)
    raise RuntimeError("server not ready")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--engine", choices=["yunshu", "yunshu-tf4", "tf-native"], required=True
    )
    ap.add_argument("--kind", choices=["code", "prose"], default="code")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--warm-tokens", type=int, default=96)
    ap.add_argument("--model")
    ap.add_argument("--dylib")
    ap.add_argument("--no-drafter", action="store_true")
    ap.add_argument("--drafter")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    tmp = Path(os.environ["TMPDIR"])
    tag = f"{a.engine}-{a.kind}-{a.ctx}-{os.getpid()}"
    home = tmp / f"home-{tag}"
    home.mkdir(parents=True, exist_ok=True)
    log = a.out.with_suffix(
        ".server.log"
    )  # next to the result, survives the job TMPDIR
    log.unlink(missing_ok=True)
    proc, url = launch(a, home, log)
    rows, rc = [], 1
    try:
        model = wait_ready(proc, url)
        text = (PROMPTS / f"{a.kind}-{a.ctx}.txt").read_text()
        for _ in range(
            a.warm
        ):  # compile / first-use costs stay out of the measured rounds
            send(
                url,
                dict(
                    model=model,
                    messages=[{"role": "user", "content": WARM_PROMPT}],
                    max_tokens=a.warm_tokens,
                    temperature=0,
                    chat_template_kwargs={"enable_thinking": False},
                ),
            )
        for rep in range(a.reps):
            r = send(
                url,
                dict(
                    model=model,
                    messages=[{"role": "user", "content": text}],
                    max_tokens=a.tokens,
                    temperature=0,
                    chat_template_kwargs={"enable_thinking": False},
                ),
            )
            if r["ct"] < a.tokens * 0.9:
                raise RuntimeError(f"short output {r['ct']}")
            r["rep"] = rep
            rows.append(r)
        time.sleep(2)  # let the server flush its per-request / per-stream log lines
        lines = log.read_text(errors="replace").splitlines()
        tf_done = [
            d
            for d in map(parse_tf_done, lines)
            if d and d["tokens"] >= a.tokens * 0.9 and d["prompt"] > 500
        ]
        for i, r in enumerate(rows):
            spec = (r.get("xy") or {}).get("speculative") or {}
            rounds = spec.get("rounds")
            if a.engine == "tf-native":
                d = tf_done[i] if i < len(tf_done) else None
                if d is None:
                    raise RuntimeError("no TensorFold round receipt")
                r["tf"] = d
                rounds = d["rounds"]
            if not rounds:
                raise RuntimeError(f"engine reported no rounds (rep {i})")
            r["summary"] = round_summary(r["ct"], rounds, r["decode_s"])
            r["spec"] = spec
        rc = 0
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(Exception):
            proc.wait(30)
        lines = log.read_text(errors="replace").splitlines() if log.exists() else []
        a.out.write_text(
            json.dumps(
                dict(
                    args={k: str(v) for k, v in vars(a).items()},
                    complete=rc == 0,
                    rows=rows,
                    log_lines=[ln for ln in lines if KEEP.search(ln)],
                    tf_lines=[ln for ln in lines if "[tensorfold] done" in ln][-8:],
                ),
                indent=1,
            )
        )
    for r in rows:
        m = r["summary"]
        print(
            f"{a.engine} {a.kind} {a.ctx} ct {r['ct']} tok/round {m['tokens_per_round']:.2f} "
            f"ms/round {m['wall_ms_per_round']:.1f} tok/s {m['tok_s']:.1f}"
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())

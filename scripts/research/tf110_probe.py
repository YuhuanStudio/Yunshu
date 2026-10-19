"""Native TensorFold copy-graft A/B and draft/plain parity; run ONLY under gpuq.

No generated text is persisted. This is a new competitor-only measurement, not a Yunshu verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

BIN = "/Volumes/P5Plus/yunshu-test-envs/bench016/TensorFold-v1.1.0/tensorfold-1.1.0-macos-arm64/bin/tensorfold-native"
MODEL = "/Volumes/P5Plus/models/TensorFold/Qwen3.8-27B-MLX-4bit"
DRAFT = "/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2"


def validate(reply: dict, count: int, drafting: bool) -> dict:
    if reply.get("choices", [{}])[0].get("finish_reason") != "length":
        raise ValueError("incomplete/short reply")
    usage = reply.get("usage", {})
    if usage.get("completion_tokens") != count:
        raise ValueError("wrong completion token count")
    rt = reply.get("runtime", {})
    # Native releases expose metadata under tensorfold, checked explicitly by CPU fixtures.
    if not rt:
        rt = reply.get("tensorfold", {})
    if rt.get("sampling") != "greedy" or rt.get("drafts") is not drafting:
        raise ValueError(
            f"wrong engaged mode; metadata keys={list(reply)} runtime keys={list(rt)}"
        )
    digest = rt.get("token_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("missing raw-token sha256")
    spec = reply.get("speculative", {})
    if not spec:
        spec = rt.get("speculative", {})
    if drafting and not spec.get("rounds"):
        raise ValueError(f"missing round stats: {list(reply)}")
    return {
        "usage": usage,
        "runtime": rt,
        "speculative": spec,
        "finish": "length",
        "token_sha256": digest,
    }


def post(port: int, body: dict) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.load(r)


def port_free() -> int:
    for port in range(18990, 19000):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free allowed server port")


def arm(args, depth: int, rep: int, copies: int, emit) -> None:
    home = args.work / f"L{depth}-r{rep}-copy{copies}"
    home.mkdir(parents=True, exist_ok=False)
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("TF_", "TENSORFOLD_", "YUNSHU_", "MLX_"))
    }
    env.update(
        HOME=str(home),
        XDG_CACHE_HOME=str(home / "cache"),
        HF_HOME=str(home / "hf"),
        TMPDIR=str(home),
        TF_Q27_COPY_ROWS=str(copies),
    )
    prompt = (args.prompts / f"code-{depth}.txt").read_text()
    port = port_free()
    argv = [
        BIN,
        "serve",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--drafter",
        DRAFT,
        "--no-update-check",
        "--no-thinking",
        "--context",
        str(depth + 4096),
    ]
    log_path = home / "server.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(900):
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"server exited {proc.returncode}; see {log_path}"
                    )
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/v1/models", timeout=2
                    ) as r:
                        model = json.load(r)["data"][0]["id"]
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(1)
            else:
                raise TimeoutError("server not ready")
            if "DFlash2 drafts for" not in log_path.read_text():
                raise ValueError("missing DFlash2 server engagement")
            body = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.tokens,
                "temperature": 0,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            expected = None
            # Cold+warm+plain, preserving actual APC state and exact token hash.
            for phase, drafting in [("cold", True), ("warm", True), ("plain", False)]:
                start = time.monotonic()
                reply = post(port, dict(body, draft=drafting))
                row = validate(reply, args.tokens, drafting)
                if row["usage"].get("prompt_tokens") != depth:
                    raise ValueError("wrong prompt depth")
                if expected is None:
                    expected = row["token_sha256"]
                if row["token_sha256"] != expected:
                    raise ValueError(
                        f"spec/APC raw-token mismatch at {depth}/{rep}/{copies}/{phase}"
                    )
                emit(
                    {
                        "depth": depth,
                        "rep": rep,
                        "copy_rows": copies,
                        "phase": phase,
                        "wall_s": time.monotonic() - start,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "server_log": str(log_path),
                        **row,
                    }
                )
            # Follow-up includes reply in memory only.
            follow = dict(body, max_tokens=min(128, args.tokens), draft=True)
            follow["messages"] += [
                {
                    "role": "assistant",
                    "content": reply["choices"][0]["message"]["content"],
                },
                {
                    "role": "user",
                    "content": "Continue the code review in full detail. Quote and rewrite the next functions. Keep writing.",
                },
            ]
            reply = post(port, follow)
            row = validate(reply, min(128, args.tokens), True)
            emit(
                {
                    "depth": depth,
                    "rep": rep,
                    "copy_rows": copies,
                    "phase": "turn2",
                    **row,
                }
            )
        finally:
            # Only the Popen child created above belongs to this probe.
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depths", nargs="+", type=int, default=[32768, 131072])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--copies", nargs="+", type=int, default=[48, 0])
    p.add_argument(
        "--prompts",
        type=Path,
        default=Path("/Volumes/P5Plus/yunshu-build/snapshot014/corpus/prompts"),
    )
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    for depth in args.depths:
        if not (args.prompts / f"code-{depth}.txt").is_file():
            p.error("prompt missing")
    if args.dry_run:
        print(json.dumps({"dry_run": True, "depths": args.depths, "arms": args.copies}))
        return
    if not os.environ.get("GPUQ_JOB_ID"):
        p.error("gpuq required")
    args.work.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as out:

        def emit(row):
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(json.dumps(row), flush=True)

        for rep in range(args.reps):
            for depth in args.depths:
                for copies in args.copies if rep % 2 == 0 else args.copies[::-1]:
                    arm(args, depth, rep, copies, emit)
        emit(
            {
                "complete": True,
                "device": "m5",
                "reps": args.reps,
                "note": "Timing qualified only after external gpuq receipt contention/swap checks",
            }
        )


if __name__ == "__main__":
    main()

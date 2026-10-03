"""Same-checkpoint HTTP comparison; run GPU portions only through gpuq."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAPID = Path("/Users/yuhuan/Documents/YuhuanStudio/Yunshu/reference/Rapid-MLX")
PYTHONS = {
    "rapid": "/Volumes/P5Plus/yunshu-test-envs/rapid-mlx/.venv/bin/python",
    "yunshu": "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python",
}


def request(url, body, raw_path=None):
    start = time.perf_counter()
    first = last = None
    text = []
    usage = None
    done = False
    raw_lines = []
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        for line in response:
            raw_lines.append(line)
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                done = True
                continue
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                token = delta.get("content") or delta.get("reasoning_content")
                if token:
                    last = time.perf_counter()
                    first = first or last
                    text.append(token)
    end = time.perf_counter()
    tokens = (usage or {}).get("completion_tokens")
    if raw_path is not None:
        Path(raw_path).write_bytes(b"".join(raw_lines))
    if not done or first is None or not tokens:
        raise RuntimeError(
            "incomplete SSE: missing DONE, first delta or completion usage"
        )
    return {
        "first_event_s": first,
        "last_event_s": last,
        "ttft_s": first - start if first else None,
        "wall_s": end - start,
        "decode_tps": (tokens - 1) / (last - first)
        if tokens and tokens > 1 and last > first
        else None,
        "usage": usage,
        "done": done,
        "output": "".join(text),
        "output_sha256": hashlib.sha256("".join(text).encode()).hexdigest(),
    }


def arm_directory(output, engine, rep, size):
    """Namespace all raw artifacts by the unique result file, not just its parent."""
    return output.parent / output.stem / f"{engine}-r{rep}-n{size}"


def run_arm(args, engine, rep, size, write):
    arm = arm_directory(args.output, engine, rep, size)
    arm.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("YUNSHU_") or key.startswith("RAPID_MLX_"):
            del env[key]
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        RAPID_MLX_TELEMETRY="0",
        DO_NOT_TRACK="1",
        RAPIDMLX_NO_UPDATE_CHECK="1",
        HF_HOME="/Volumes/P5Plus/hf-cache",
        UV_CACHE_DIR="/Volumes/P5Plus/yunshu-test-envs/-cache",
        XDG_CACHE_HOME=str(arm / "cache"),
        TMPDIR=str(arm / "tmp"),
        PYTHONPATH=str(ROOT / "python"),
    )
    (arm / "tmp").mkdir(exist_ok=True)
    # Preserve Yunshu's bounded internal APC product default; Rapid's unrelated
    # user state and run scratch stay external. Override HOME only for Rapid.
    if engine == "rapid":
        env["HOME"] = str(arm / "home")
        Path(env["HOME"]).mkdir(exist_ok=True)
        cmd = [
            PYTHONS[engine],
            "-m",
            "rapid_mlx.cli",
            "serve",
            args.model,
            "--port",
            "18998",
        ] + args.rapid_flags
    else:
        cmd = [
            PYTHONS[engine],
            "-m",
            "yunshu_cli",
            "serve",
            "--model",
            args.model,
            "--port",
            "18998",
        ]
    log = arm / "server.log"
    with log.open("w") as out:
        startup_started = time.perf_counter()
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        peak = [0]
        stop = threading.Event()

        def sample_memory():
            import psutil

            while not stop.wait(0.1):
                try:
                    parent = psutil.Process(process.pid)
                    rss = sum(
                        p.memory_info().rss
                        for p in [parent, *parent.children(recursive=True)]
                    )
                    peak[0] = max(peak[0], rss)
                except psutil.Error:
                    pass

        monitor = threading.Thread(target=sample_memory, daemon=True)
        monitor.start()
        try:
            url = "http://127.0.0.1:18998"
            for _ in range(300):
                if process.poll() is not None:
                    raise RuntimeError(f"server exited {process.returncode}: {log}")
                try:
                    with urllib.request.urlopen(
                        url + "/v1/models", timeout=2
                    ) as response:
                        models = json.load(response)
                    if models.get("data"):
                        with urllib.request.urlopen(
                            url + "/health/ready", timeout=2
                        ) as response:
                            if response.status == 200:
                                break
                except Exception:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError(f"server not ready: {log}")
            write(
                dict(
                    engine=engine,
                    rep=rep,
                    size=size,
                    case="startup",
                    ready_s=time.perf_counter() - startup_started,
                )
            )
            model_id = models["data"][0]["id"]
            # Tokenize locally, never load model weights on this client.
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
            seed = tokenizer.encode(
                "The archive entry has code ALPHA and status OPEN.\n",
                add_special_tokens=False,
            )
            prefix = tokenizer.decode((seed * (size // len(seed) + 1))[:size])
            # A unique shared prefix per repetition prevents persisted Yunshu
            # APC from turning a nominal fresh-process cold request into a hit.
            nonce = hashlib.sha256(
                f"{getattr(args, 'prompt_identity', args.output.resolve())}:{rep}:{size}".encode()
            ).hexdigest()[:16]
            prefix = f"Run identity {nonce}.\n" + prefix
            base = {
                "model": model_id,
                "temperature": 0,
                "max_tokens": args.tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [
                    {
                        "role": "user",
                        "content": prefix
                        + "\nList integers from 1 to 1000, comma separated.",
                    }
                ],
            }
            previous = None
            for case in ["cold", "warm", "turn2"]:
                body = json.loads(json.dumps(base))
                if case == "turn2":
                    body["messages"] += [
                        {"role": "assistant", "content": previous["output"]},
                        {"role": "user", "content": "Continue listing integers."},
                    ]
                (arm / f"{case}.request.json").write_text(json.dumps(body))
                row = request(url, body, arm / f"{case}.sse")
                previous = row
                write(
                    dict(
                        engine=engine,
                        rep=rep,
                        size=size,
                        case=case,
                        command=cmd,
                        server_log=str(log),
                        **row,
                    )
                )
            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                bodies = []
                for i in range(8):
                    body = json.loads(json.dumps(base))
                    body["messages"][0]["content"] = (
                        f"Concurrent request {i}.\n" + body["messages"][0]["content"]
                    )
                    bodies.append(body)
                rows = list(
                    pool.map(
                        lambda item: request(
                            url, item[1], arm / f"concurrent-{item[0]}.sse"
                        ),
                        enumerate(bodies),
                    )
                )
            wall = time.perf_counter() - start
            write(
                dict(
                    engine=engine,
                    rep=rep,
                    size=size,
                    case="concurrent8",
                    wall_s=wall,
                    aggregate_decode_tps=sum(
                        r["usage"]["completion_tokens"] - 1 for r in rows
                    )
                    / (
                        max(r["last_event_s"] for r in rows)
                        - min(r["first_event_s"] for r in rows)
                    ),
                    aggregate_e2e_tps=sum(
                        (r["usage"] or {}).get("completion_tokens", 0) for r in rows
                    )
                    / wall,
                    requests=rows,
                )
            )
            try:
                memory = subprocess.run(
                    ["footprint", "-p", str(process.pid)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                (arm / "footprint.txt").write_text(memory.stdout + memory.stderr)
                match = re.search(
                    r"phys_footprint_peak:\s+([0-9.]+)\s+(MB|GB)", memory.stdout
                )
                write(
                    dict(
                        engine=engine,
                        rep=rep,
                        size=size,
                        case="physical_memory",
                        peak_gib=float(match.group(1))
                        / (1024 if match.group(2) == "MB" else 1)
                        if match
                        else None,
                        rc=memory.returncode,
                        raw=str(arm / "footprint.txt"),
                    )
                )
            except Exception as exc:
                write(
                    dict(
                        engine=engine,
                        rep=rep,
                        size=size,
                        case="physical_memory",
                        error=repr(exc),
                    )
                )
            if args.tool_eval and rep == 0 and size == args.sizes[0]:
                quality = arm / "rapid-tool-eval.json"
                result = subprocess.run(
                    [
                        PYTHONS["rapid"],
                        str(RAPID / "evals/run_eval.py"),
                        "--model",
                        f"{engine}-same-checkpoint",
                        "--port",
                        "18998",
                        "--suite",
                        "tool_calling",
                        "--model-path",
                        args.model,
                        "--engine",
                        "batched",
                        "--output",
                        str(quality),
                    ],
                    env=env,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                )
                write(
                    dict(
                        engine=engine,
                        rep=rep,
                        size=size,
                        case="rapid_tool_eval",
                        rc=result.returncode,
                        result=str(quality),
                        exists=quality.is_file(),
                    )
                )
                if result.returncode or not quality.is_file():
                    raise RuntimeError("Rapid tool evaluation did not complete")
                from agent_shapes import run as run_agent_shapes

                replay_dir = arm / "census-replay"
                replay = subprocess.run(
                    [
                        PYTHONS["yunshu"],
                        str(ROOT / "scripts/research/agent_compat/replay.py"),
                        "--url",
                        url,
                        "--model",
                        args.model,
                        "--sessions",
                        "cc_plain,cc_bash_edit,cx_plain,cx_shell",
                        "--max-tokens",
                        "256",
                        "--out",
                        str(replay_dir),
                    ],
                    env=env,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                )
                write(
                    dict(
                        engine=engine,
                        rep=rep,
                        size=size,
                        case="census_replay",
                        rc=replay.returncode,
                        result=str(replay_dir / "replay.json"),
                        exists=(replay_dir / "replay.json").is_file(),
                    )
                )
                if replay.returncode or not (replay_dir / "replay.json").is_file():
                    raise RuntimeError("Yunshu census replay did not complete")
                shape_result = run_agent_shapes(url, model_id)
                (arm / "agent-shapes.json").write_text(
                    json.dumps(shape_result, indent=2)
                )
                write(
                    dict(
                        engine=engine,
                        rep=rep,
                        size=size,
                        case="agent_shapes",
                        passed=shape_result["passed"],
                        total=shape_result["total"],
                        result=str(arm / "agent-shapes.json"),
                    )
                )
        finally:
            mode_lines = [
                line
                for line in log.read_text(errors="replace").splitlines()
                if any(
                    word in line.lower()
                    for word in (
                        "speculative",
                        "mtp",
                        "dflash",
                        "backend",
                        "hybrid",
                        "prefix cache",
                    )
                )
            ]
            write(
                dict(
                    engine=engine,
                    rep=rep,
                    size=size,
                    case="engaged_mode",
                    evidence=mode_lines[-40:],
                )
            )
            stop.set()
            monitor.join(timeout=2)
            write(
                dict(
                    engine=engine,
                    rep=rep,
                    size=size,
                    case="memory",
                    peak_process_tree_rss_bytes=peak[0],
                    memory_method="100ms process-tree RSS sampling; not Metal allocation peak",
                )
            )
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1024, 8192, 32768])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument(
        "--engines", nargs="+", choices=list(PYTHONS), default=list(PYTHONS)
    )
    parser.add_argument("--tool-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rapid-flags", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    if not (Path(args.model) / "config.json").is_file():
        parser.error("model must be a local checkpoint with config.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def write(row):
        with args.output.open("a") as out:
            out.write(json.dumps(row) + "\n")
        print(
            json.dumps(
                {k: v for k, v in row.items() if k not in ("output", "requests")}
            ),
            flush=True,
        )

    if args.dry_run:
        write(
            {
                "complete": True,
                "dry_run": True,
                "engines": args.engines,
                "model": args.model,
            }
        )
        return
    failures = 0
    for rep in range(args.reps):
        for size in args.sizes:
            for engine in (
                args.engines if rep % 2 == 0 else list(reversed(args.engines))
            ):
                try:
                    run_arm(args, engine, rep, size, write)
                except Exception as exc:
                    failures += 1
                    write(
                        {"engine": engine, "rep": rep, "size": size, "error": repr(exc)}
                    )
    write({"complete": True, "failures": failures})
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()

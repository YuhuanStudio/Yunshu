#!/usr/bin/env python3
"""Request-scoped round attribution for the installed Kapitsa greedy protocol.

Stdlib only. Run model work through gpuq. Stores metrics and hashes, never text.
The external harness supplies the exact request and middle-window rate algorithm.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import random
import re
import signal
import socket
import subprocess
import time
from pathlib import Path

ROUND = re.compile(
    r"DFlash(?: chain)? round: at=(\d+) kind=(\w+) rows=(\d+) accepted=(\d+) match=(-?\d+) ms=([\d.]+)"
)


def summarize_rounds(log):
    rounds = [
        dict(at=int(a), kind=k, rows=int(r), accepted=int(n), match=int(m), ms=float(t))
        for a, k, r, n, m, t in ROUND.findall(log)
    ]
    kinds = {}
    for kind in sorted({r["kind"] for r in rounds}):
        group = [r for r in rounds if r["kind"] == kind]
        count = len(group)
        tokens = sum(r["accepted"] for r in group)
        elapsed = sum(r["ms"] for r in group)
        kinds[kind] = dict(
            rounds=count,
            tokens=tokens,
            tokens_per_round=tokens / count,
            ms_per_round=elapsed / count,
            total_ms=elapsed,
            mean_rows=sum(r["rows"] for r in group) / count,
        )
    return dict(rounds=rounds, by_kind=kinds, round_count=len(rounds))


def competitor_rounds(log, engine, completion_tokens):
    """Keep native counter definitions: mlx-serve accepts excludes each round bonus."""
    if engine == "mlxserve":
        found = re.findall(
            r"\[spec-stats\] mode=dflash attempts=(\d+) accepts=(\d+)", log
        )
        if found:
            attempts, accepts = map(int, found[-1])
            return dict(
                rounds=attempts,
                accepted_drafts=accepts,
                tokens_per_round=completion_tokens / attempts if attempts else None,
                counter_source="spec-stats; accepts excludes bonus",
            )
    if engine == "kapitsa":
        found = re.findall(
            r"chat: prompt (\d+).*?, (\d+) tokens in ([\d.]+) ms .*?, ([\d.]+) tok/cycle",
            log,
        )
        if found:
            prompt, tokens, elapsed, tpr = found[-1]
            cycles = round((int(tokens) - 1) / float(tpr))
            return dict(
                prompt_tokens=int(prompt),
                rounds=cycles,
                tokens_per_round=float(tpr),
                ms_per_round=float(elapsed) / cycles,
                counter_source="request chat summary; rounded tok/cycle",
            )
    return None


def safe_result(result):
    result = dict(result)
    text = result.pop("text", "") or ""
    reasoning = result.pop("reasoning", "") or ""
    result["text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    result["reasoning_chars"] = len(reasoning)
    return result


def validate(result, summary, engine, *, require_fast=True):
    if result.get("invalid") or result.get("completion_tokens") != 512:
        raise RuntimeError(f"invalid request: {result.get('invalid')}")
    if (
        engine == "yunshu"
        and require_fast
        and not summary.get("fast_round_count", summary["round_count"])
    ):
        raise RuntimeError("fast tree not engaged: no request-scoped DFlash round logs")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine", choices=("yunshu", "kapitsa", "mlxserve"), default="yunshu"
    )
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--extra-body", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-chain", action="store_true")
    parser.add_argument("--plain", action="store_true")
    parser.add_argument("--repeat-exact", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    competitors = Path("/Volumes/P5Plus/yunshu-build/competitors")
    bench = load_module("kapitsa_bench", competitors / "kapitsa/bench/harness/bench.py")
    launcher = load_module("competitor_launcher", competitors / "harness/runjob.py")
    args.out.mkdir(parents=True, exist_ok=True)
    engine = "yunshu-oq4e" if args.engine == "yunshu" else args.engine
    cmd, env, port = launcher.launch(engine, str(args.out))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if args.engine == "kapitsa":
        env["NI_LOG"] = "2"
    if args.engine == "yunshu":
        cmd = [
            launcher.PY,
            str(Path(__file__).with_name("shortdecode_chain_trace.py")),
            *cmd[1:],
        ]
        env.update(
            PYTHONPATH=str(args.source / "python"),
            YUNSHU_LOG_LEVEL="DEBUG",
            SHORTDECODE_PLAIN="1" if args.plain else "0",
            YUNSHU_VLM_APC_DISK_DIR=f"/Volumes/Micron/yunshu-apc/{args.out.name}",
        )
    # Reserve only the documented pool, fail before loading weights if occupied.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))
    a = bench.parse_args(
        [
            "--base-url",
            f"http://127.0.0.1:{port}/v1",
            "--model",
            "pending",
            "--label",
            args.out.name,
            "--extra-body",
            args.extra_body,
        ]
    )
    random.seed(1021)
    if args.dry_run:
        body = bench.body_of(a, bench.decode_messages(), 512, greedy=True)
        print(
            json.dumps(dict(cmd=cmd, env=env, request_keys=sorted(body), complete=True))
        )
        return
    manifest = dict(
        source=str(args.source),
        head=subprocess.check_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
        ).strip(),
        tracked_diff_sha256=hashlib.sha256(
            subprocess.check_output(["git", "-C", str(args.source), "diff", "HEAD"])
        ).hexdigest(),
        command=cmd,
        server_env=env,
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log = args.out / "server.log"
    with log.open("wb") as stream:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        model, ready_s = launcher.wait_ready(proc, port, str(log))
        a.model = model
        for i in range(3):
            subprocess.run(
                [
                    launcher.PY,
                    "-I",
                    "-c",
                    launcher.WARM,
                    f"http://127.0.0.1:{port}",
                    model,
                    str(i),
                ],
                check=True,
            )
        if args.engine == "kapitsa":
            time.sleep(75)
        random.seed(1021)
        with (args.out / "metrics.jsonl").open("w") as output:
            for i in range(args.runs):
                if args.repeat_exact:
                    random.seed(1021)
                offset = log.stat().st_size
                result = safe_result(bench.t_decode(a, greedy=True))
                # Server summary can follow the last SSE event by a scheduling tick.
                time.sleep(0.1)
                with log.open("rb") as source:
                    source.seek(offset)
                    request_log = source.read().decode(errors="replace")
                summary = summarize_rounds(request_log)
                receipts = re.findall(
                    r"Raw token receipt: count=(\d+) sha256=([a-f0-9]+)", request_log
                )
                summary["raw_receipts"] = receipts
                if args.engine == "yunshu" and (
                    len(receipts) != 1 or int(receipts[0][0]) != 512
                ):
                    raise RuntimeError(f"missing raw token receipt: {receipts}")
                summary["competitor"] = competitor_rounds(
                    request_log, args.engine, result.get("completion_tokens", 0)
                )
                summary["fast_round_count"] = request_log.count("DFlash round: at=")
                summary["chain_round_count"] = request_log.count(
                    "DFlash chain round: at="
                )
                record = dict(
                    engine=args.engine, rep=i, ready_s=ready_s, result=result, **summary
                )
                (args.out / f"request-{i}.log").write_text(request_log)
                output.write(json.dumps(record) + "\n")
                output.flush()
                validate(
                    result, summary, args.engine, require_fast=not args.allow_chain
                )
                print(
                    json.dumps({k: v for k, v in record.items() if k != "rounds"}),
                    flush=True,
                )
            output.write(
                json.dumps(dict(complete=True, engine=args.engine, requests=args.runs))
                + "\n"
            )
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


if __name__ == "__main__":
    main()

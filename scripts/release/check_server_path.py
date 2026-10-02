"""Guard: the server must decode as fast as the engine does in-process.

Loads the model in this process (``VLMEngine``, repo code) and, for the same
prompts, compares single-request greedy decode through the batch runner
directly against a running server's streaming ``/v1/chat/completions``:

- same prompt, same ``max_tokens``, greedy, thinking off;
- the server must return the same text as the in-process tokens decode to
  (otherwise the comparison is not apples to apples and the check fails);
- decode tok/s = tokens after the first / time after the first (the server's
  token count comes from ``usage``); one untimed warm-up run per case, then
  the median of ``--repeats``;
- ratio = server / in-process per case; FAIL when the worst case is below
  ``--min-ratio`` (default 0.95).

Run it while the server is idle (the two share the GPU):

    python scripts/release/check_server_path.py --url http://127.0.0.1:18764 \\
        --model <ckpt> --output runs/server-path-guard.json

Prints one JSON summary line; exit 0 = PASS, 1 = FAIL, 2 = could not measure,
3 = CONTENDED. --retry-contended waits for a bounded quiet CPU window and retries
once, preserving both attempts. Contention cannot excuse text/spec mismatches;
a quiet ratio failure still fails. Two contended attempts require remeasurement.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "dev"))
from gpuq_contention import (  # noqa: E402
    server_path_attempts,
    wait_for_quiet,
    was_contended,
)

sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))


def inprocess_run(engine, msgs, max_tokens: int) -> tuple[float | None, str]:
    ids, pkw, salt = engine._executor.submit(
        engine._runner_input, msgs, [], [], False, {}
    ).result()
    runner = engine._batch_runner
    times: list[float] = []
    tokens: list[int] = []

    def consume():
        for tok in runner.iter_tokens(
            ids,
            max_tokens=max_tokens,
            temperature=0.0,
            prompt_kwargs=pkw,
            apc_semantic_hash=salt,
        ):
            times.append(time.perf_counter())
            tokens.append(tok)

    th = threading.Thread(target=consume)
    th.start()
    th.join()
    eos = set(engine._get_eos_ids())
    text = engine._tokenizer.decode([t for t in tokens if t not in eos])
    span = times[-1] - times[0] if len(times) > 1 else 0.0
    return ((len(times) - 1) / span if span else None), text


def _inprocess_spec_kind(engine) -> str:
    """The spec method the in-process engine selected (same code as the server)."""
    from yunshu_engine import spec_select
    from yunshu_engine.mlxvlm_mtp import is_mtp_capable

    choice = spec_select.choose(
        engine._config, spec_family=True, mtp_capable=is_mtp_capable(engine._model_path)
    )
    return choice.kind


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", required=True)
    ap.add_argument(
        "--model", required=True, help="checkpoint dir the server is serving"
    )
    ap.add_argument("--served-name", default="Qwen3.8-27B")
    ap.add_argument("--tasks", nargs="+", default=["code", "prose"])
    ap.add_argument("--contexts", type=int, nargs="+", default=[0, 8192])
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--min-ratio", type=float, default=0.95)
    ap.add_argument("--output", type=Path)
    ap.add_argument(
        "--server-log",
        type=Path,
        help="server log; its 'Speculative decoding: <kind>' line must match "
        "the in-process selection (same env => same path)",
    )
    ap.add_argument(
        "--retry-contended",
        action="store_true",
        help="retry once after a bounded quiet CPU wait",
    )
    a = ap.parse_args()
    if a.retry_contended:
        summary = server_path_attempts(
            lambda: measure(a), before_retry=lambda: wait_for_quiet(os.getpgrp())
        )
    else:
        summary = measure(a)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return {"PASS": 0, "FAIL": 1, "CONTENDED": 3}.get(summary["status"], 2)


def measure(a):
    from probe_server_path import messages_for, run_http

    started = time.time()
    from yunshu_engine.vlm_engine import VLMEngine

    engine = VLMEngine(a.model)
    loop = asyncio.new_event_loop()
    loop.run_until_complete(engine.start())
    cases = []
    spec, spec_ok = None, True
    if a.server_log:
        import re

        m = re.findall(r"Speculative decoding: (\w+)", a.server_log.read_text())
        srv = m[-1] if m else "none"
        loc = _inprocess_spec_kind(engine)
        spec = f"server={srv} inprocess={loc}"
        spec_ok = srv == loc
    try:
        for ctx in a.contexts:
            for task in a.tasks:
                msgs = messages_for(task, ctx)
                inproc_text = inprocess_run(engine, msgs, a.max_tokens)[1]  # warm-up
                run_http(a.url, a.served_name, msgs, a.max_tokens)  # warm-up
                ip, sv, same, diffs = [], [], True, []
                for _ in range(a.repeats):
                    rate, text = inprocess_run(engine, msgs, a.max_tokens)
                    ip.append(rate or 0.0)
                    r = run_http(a.url, a.served_name, msgs, a.max_tokens)
                    sv.append(r["decode_tps"] or 0.0)
                    for who, t in (("inprocess", text), ("server", r["text"])):
                        if t != inproc_text:
                            k = next(
                                (
                                    i
                                    for i, (x, y) in enumerate(zip(t, inproc_text))  # noqa: B905 - Python 3.9
                                    if x != y
                                ),
                                min(len(t), len(inproc_text)),
                            )
                            diffs.append(
                                {
                                    "side": who,
                                    "first_diff_char": k,
                                    "len": [len(t), len(inproc_text)],
                                    "got": t[max(0, k - 20) : k + 40],
                                    "ref": inproc_text[max(0, k - 20) : k + 40],
                                }
                            )
                    same = same and text == inproc_text and r["text"] == inproc_text
                ip_m, sv_m = statistics.median(ip), statistics.median(sv)
                cases.append(
                    {
                        "task": task,
                        "context": ctx,
                        "inprocess_tps": round(ip_m, 2),
                        "server_tps": round(sv_m, 2),
                        "ratio": round(sv_m / ip_m, 3) if ip_m else None,
                        "same_text": same,
                        "diffs": diffs,
                    }
                )
    finally:
        loop.run_until_complete(engine.stop())
    ratios = [c["ratio"] for c in cases if c["ratio"] is not None]
    worst = min(ratios) if ratios else None
    all_same = all(c["same_text"] for c in cases)
    status = (
        "PASS"
        if worst is not None and worst >= a.min_ratio and all_same and spec_ok
        else "FAIL"
    )
    if worst is None:
        status = "ERROR"
    summary = {
        "status": status,
        "contended": was_contended(started, time.time()),
        "started": started,
        "ended": time.time(),
        "worst_ratio": worst,
        "min_ratio": a.min_ratio,
        "same_text": all_same,
        "spec": spec,
        "spec_match": spec_ok,
        "cases": cases,
    }
    return summary


if __name__ == "__main__":
    sys.exit(main())

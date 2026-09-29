"""Does background ANE / CPU work steal GPU decode throughput (shared LPDDR5X)?

GPU decode proxy (27B-like ~14 GB/step, pipelined) alone; the background load alone; then
both together. Reports decode step time change and background call-rate change.

    PYTHONPATH=scripts/research/hw python scripts/research/hw/bg_gpu_concurrency.py \
        --kind ane --model /Volumes/P5Plus/yunshu-test-cache/ane/layer_x1_M8_fp16.mlpackage
    ... --kind cpu_gemv --procs 4
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

from _common import Out
from _proxy import DecodeProxy, stats

HERE = Path(__file__).parent
ANE_PY = "/Volumes/P5Plus/yunshu-test-envs/ane/bin/python"
MAIN_PY = "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python"


def spawn(kind, model, procs):
    py = ANE_PY if kind == "ane" else MAIN_PY
    ps = []
    for _ in range(procs):
        cmd = [py, str(HERE / "_bg_worker.py"), "--kind", kind]
        if model:
            cmd += ["--model", model]
        p = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        ps.append(p)
    for p in ps:
        assert p.stdout.readline().strip() == "READY"
    return ps


def finish(ps):
    calls = []
    for p in ps:
        p.stdin.write("stop\n")
        p.stdin.flush()
        calls += json.loads(p.stdout.readline())
        p.wait()
    return calls


def rate(calls, t0, t1):
    inside = [(a, b) for a, b in calls if a >= t0 and b <= t1]
    if not inside:
        return {}
    d = sorted(b - a for a, b in inside)
    return {
        "calls_per_s": round(len(inside) / (t1 - t0), 1),
        "ms_median": round(d[len(d) // 2] * 1e3, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True)
    ap.add_argument("--model")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument(
        "--tokens-per-call",
        type=int,
        default=0,
        help="prompt tokens one ANE call covers (reports background tok/s)",
    )
    ap.add_argument("--seconds", type=float, default=10)
    a = ap.parse_args()
    out = Out("bg_gpu_concurrency")
    label = {
        "bg_kind": a.kind,
        "procs": a.procs,
        "model": Path(a.model).name if a.model else None,
    }

    dec = DecodeProxy()
    dec.run(2)  # warm
    alone = stats(dec.run(a.seconds))

    ps = spawn(a.kind, a.model, a.procs)
    t0 = time.time()
    time.sleep(a.seconds)
    t1 = time.time()
    bg_alone = rate(finish(ps), t0 + 1, t1)

    ps = spawn(a.kind, a.model, a.procs)
    time.sleep(1)
    t0 = time.time()
    both = stats(dec.run(a.seconds))
    t1 = time.time()
    bg_both = rate(finish(ps), t0, t1)

    if a.tokens_per_call:
        label["tokens_per_call"] = a.tokens_per_call
        for r in (bg_alone, bg_both):
            r["tok_s"] = round(r.get("calls_per_s", 0) * a.tokens_per_call, 1)
    out(
        kind="concurrency",
        **label,
        gpu_alone=alone,
        gpu_with_bg=both,
        bg_alone=bg_alone,
        bg_with_gpu=bg_both,
        gpu_tok_s_ratio=round(both["tok_s"] / alone["tok_s"], 3),
        bg_rate_ratio=round(
            bg_both.get("calls_per_s", 0) / max(bg_alone.get("calls_per_s", 1), 1e-9), 3
        ),
    )


if __name__ == "__main__":
    main()

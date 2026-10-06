"""Run scripts/verify/verify_*.py that no gate or yv stage references, one subprocess each (coverage audit).

Prints one line per script (rc, seconds, last output line) and exits nonzero when any fails or
times out (fail closed). GPU work: submit through gpuq.

    python covaudit_orphans.py --model PATH [--only a,b] [--timeout 300]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = [
    "context_overflow",
    "kv_quant",
    "quant_kv_prefix",
    "streaming_tool_calls",
    "grammar_constraints",
    "structured_output",
    "anthropic_tools",
    "anthropic_messages",
    "multiturn",
    "usage_accounting",
    "concurrent",
    "error_contract",
    "responses_api",
    "reasoning_parser",
    "json_mode",
]


def summarize(rc: int | None, out: str) -> str:
    last = [ln for ln in out.strip().splitlines() if ln.strip()][-1:] or [""]
    return f"{'TIMEOUT' if rc is None else 'rc=' + str(rc)} | {last[0][:140]}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--only")
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args(argv)
    names = a.only.split(",") if a.only else DEFAULT
    bad = 0
    for n in names:
        path = ROOT / "scripts/verify" / f"verify_{n}.py"
        if not path.exists():
            print(f"{n}: MISSING")
            bad += 1
            continue
        env = dict(
            os.environ,
            YUNSHU_BENCH_MODEL=a.model,
            PYTHONPATH=f"{ROOT}:{ROOT / 'python'}",
        )
        t0 = time.monotonic()
        try:
            p = subprocess.run(
                [sys.executable, str(path)],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=a.timeout,
            )
            rc, out = p.returncode, p.stdout + p.stderr
        except subprocess.TimeoutExpired as e:
            rc, out = None, str(e.stdout or "")
        print(f"{n}: {summarize(rc, out)} ({time.monotonic() - t0:.0f}s)", flush=True)
        if rc != 0:
            bad += 1
            Path(f"/Volumes/P5Plus/yunshu-build/covaudit/orphan-{n}.log").write_text(
                out
            )
    print("RESULT", "FAIL" if bad else "PASS", f"{bad} failing of {len(names)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

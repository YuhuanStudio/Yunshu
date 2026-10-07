"""Where does a warm (full APC hit) TTFT go?  One server, one long prompt, cold then warm xN.

Starts the served process through tfbench's Srv with a stack sampler injected (sitecustomize),
sends the prompt, and prints for each warm request the client TTFT plus a timeline of what the
server's threads were doing between request send and first token (runs of one stack >= 5 ms).
Usage (via gpuq): ttft_warm_probe.py --model M --kind code --ctx 32768 --warm 3 --tag base
"""

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def timeline(samples, t_lo, t_hi, min_ms=5.0):
    """Collapse samples of each thread in [t_lo, t_hi] to runs of an identical top-4 stack.
    Returns {thread: [(start_offset_ms, dur_ms, stack_str)]} for runs >= min_ms."""
    by_thread = {}
    for t, name, stack in samples:
        if t_lo <= t <= t_hi:
            by_thread.setdefault(name, []).append((t, " < ".join(stack[:4])))
    out = {}
    for name, rows in by_thread.items():
        rows.sort()
        runs = []
        cur = None
        for t, st in rows:
            if cur and cur[2] == st:
                cur[1] = t
            else:
                if cur:
                    runs.append(cur)
                cur = [t, t, st]
        if cur:
            runs.append(cur)
        out[name] = [
            (round((a - t_lo) * 1e3, 1), round((b - a) * 1e3 + 2, 1), st)
            for a, b, st in runs
            if (b - a) * 1e3 + 2 >= min_ms
        ]
    return out


def signal_dump(pid):
    """SIGUSR1 every process in pid's group (the served worker may be a child of the launcher)."""
    out = subprocess.run(
        ["pgrep", "-g", str(os.getpgid(pid))], capture_output=True, text=True
    ).stdout
    for p in out.split():
        with contextlib.suppress(ProcessLookupError):
            os.kill(int(p), signal.SIGUSR1)


def load_samples(prefix):
    rows = []
    for p in Path(prefix).parent.glob(Path(prefix).name + ".*"):
        rows += [json.loads(line) for line in p.read_text().splitlines() if line]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", default="code")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--src", required=True, help="python/ dir of the tree under test")
    ap.add_argument("--max-tokens", type=int, default=16)
    a = ap.parse_args()
    os.environ["TFB_YUNSHU_SRC"] = a.src
    os.environ["TFB_OUT"] = f"/Volumes/P5Plus/yunshu-build/ttft-probe/{a.tag}"
    os.environ["TFB_PORT_LAST"] = "18999"
    sys.path.insert(0, str(HERE))
    import tfbench

    out = Path(os.environ["TFB_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    prof = str(out / "samples")
    real_popen = subprocess.Popen

    def popen(cmd, **kw):
        if not isinstance(kw.get("env"), dict) or "serve" not in cmd:
            return real_popen(cmd, **kw)  # not the served process (pgrep etc.)
        env = dict(kw["env"])
        env["PYTHONPATH"] = f"{HERE / 'ttft_probe_site'}:{env.get('PYTHONPATH', '')}"
        env["TTFT_PROBE_OUT"] = prof
        kw["env"] = env
        return real_popen(cmd, **kw)

    tfbench.subprocess.Popen = popen
    s = tfbench.Srv("yunshu", {}, f"-probe-{a.tag}", model=a.model)
    try:
        prompt = tfbench.load_prompt(f"{a.kind}-{a.ctx}")
        rows = []
        for i in range(1 + a.warm):
            t0 = time.perf_counter()
            r = tfbench.send(s.url, tfbench.req(s.model, prompt, a.max_tokens))
            time.perf_counter()
            r.pop("_text", None)
            ttft = r["ttft_s"]
            if i == 0 and r["cached"] != 0:
                print(f"FAIL: first request not cold, cached={r['cached']}")
                sys.exit(1)
            if i > 0 and r["cached"] < r["pt"] - 8:
                print(f"FAIL: warm request {i} cached {r['cached']} of {r['pt']}")
                sys.exit(1)
            rows.append((i, t0, t0 + ttft, ttft, r["cached"], r["pt"]))
            print(
                f"req {i}: ttft {ttft:.3f}s cached {r['cached']}/{r['pt']}", flush=True
            )
        signal_dump(s.proc.pid)
        time.sleep(3)
        samples = load_samples(prof)
        if not samples:
            print("FAIL: no samples")
            sys.exit(1)
        for i, t0, tf, ttft, _cached, _pt in rows[1:]:
            print(f"== warm {i} ttft {ttft:.3f}s")
            for name, runs in timeline(samples, t0, tf).items():
                for off, dur, st in runs:
                    print(f"  [{name}] +{off:8.1f}ms {dur:7.1f}ms  {st}")
    finally:
        s.kill()


if __name__ == "__main__":
    main()

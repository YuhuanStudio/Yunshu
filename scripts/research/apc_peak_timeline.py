"""Fine-grained footprint timeline over the long-turn prefill end and the follow-up turn.

Each arm starts a fresh server from its tree and runs the memory_ab sequence (short, 8K, 32K,
96K turn 1 + turn 2). Around every request a sidecar samples the server process-tree
physical footprint every ~20 ms and /metrics (MLX active / cache / peak, APC resident bytes)
every ~100 ms. For every request it records the footprint peak, when it happened (seconds from
the request's end) and the MLX gauges nearest to it, so the part of the peak that is MLX
active memory, MLX allocator cache or neither can be told apart.

    python scripts/research/apc_peak_timeline.py --arm base=/tree --arm cand=/tree \
        --model /path/model --reps 1 --out out.jsonl
"""

import argparse
import contextlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from memory_ab import GIB, code_doc, metrics  # noqa: E402
from process_memory import process_tree_memory  # noqa: E402

DEFAULT_SIZES = (8192, 32768, 98304)


def summarize(fp_samples, gauge_samples, t_end, window=1.0):
    """Peak footprint of a request and the MLX gauges around it.

    fp_samples: [(t, bytes)], gauge_samples: [(t, {name: GiB})], t_end: request end time.
    """
    if not fp_samples:
        return None
    t_peak, peak = max(fp_samples, key=lambda s: s[1])
    near = [g for t, g in gauge_samples if abs(t - t_peak) <= window]
    nearest = min(gauge_samples, key=lambda s: abs(s[0] - t_peak), default=(None, {}))[
        1
    ]
    gauges = {}
    for key in ("active", "cache", "peak", "apc"):
        values = [g[key] for g in near if key in g]
        gauges[key + "_max_near"] = round(max(values), 3) if values else None
        gauges[key + "_nearest"] = nearest.get(key)
    tail = [b for t, b in fp_samples if t >= t_end - 1.0]
    return dict(
        peak_gib=round(peak / GIB, 3),
        t_peak_before_end=round(t_end - t_peak, 3),
        end_gib=round(tail[-1] / GIB, 3) if tail else None,
        samples=len(fp_samples),
        **gauges,
    )


def gauge_row(raw):
    """Short names for the metrics() keys."""
    out = {}
    for key, value in raw.items():
        if 'type="active"' in key:
            out["active"] = value
        elif 'type="cache"' in key:
            out["cache"] = value
        elif 'type="peak"' in key:
            out["peak"] = value
        elif "apc_resident_bytes" in key:
            out["apc"] = value
    return out


class Sidecar:
    def __init__(
        self, pid, fp_reader=None, gauge_reader=None, fp_period=0.02, gauge_period=0.1
    ):
        self.pid = pid
        self.fp_reader = fp_reader
        self.gauge_reader = gauge_reader
        self.fp_period = fp_period
        self.gauge_period = gauge_period
        self.fp = []
        self.gauges = []
        self.stop = threading.Event()
        self.threads = []

    def start(self):
        for target in (self._fp_loop, self._gauge_loop):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self.threads.append(thread)

    def _fp_loop(self):
        while not self.stop.is_set():
            with contextlib.suppress(Exception):
                self.fp.append((time.time(), self.fp_reader()))
            time.sleep(self.fp_period)

    def _gauge_loop(self):
        while not self.stop.is_set():
            with contextlib.suppress(Exception):
                self.gauges.append((time.time(), gauge_row(self.gauge_reader())))
            time.sleep(self.gauge_period)

    def take(self, t0, t1):
        fp = [(t, b) for t, b in self.fp if t0 <= t <= t1]
        g = [(t, v) for t, v in self.gauges if t0 - 1 <= t <= t1 + 1]
        return fp, g

    def close(self):
        self.stop.set()
        for thread in self.threads:
            thread.join(5)


def make_fp_reader(pid):
    """Fast footprint reader: the process tree is resolved every 2 s, usage read per sample."""
    import ctypes

    from process_memory import RusageInfoV2

    lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    fn = lib.proc_pid_rusage
    fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    state = {"t": 0.0, "pids": [pid]}

    def read():
        now = time.time()
        if now - state["t"] > 2.0:
            rows = process_tree_memory(pid).get("processes", [])
            pids = [r["pid"] for r in rows] or [pid]
            state.update(t=now, pids=pids)
        total = 0
        for p in state["pids"]:
            info = RusageInfoV2()
            if fn(p, 2, ctypes.byref(info)) == 0:
                total += info.phys_footprint
        return total

    return read


def chat(url, messages, max_tokens):
    body = json.dumps(
        dict(model="m", messages=messages, max_tokens=max_tokens, temperature=0)
    ).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions", body, {"Content-Type": "application/json"}
    )
    data = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    msg = data["choices"][0]["message"]
    return msg.get("content") or "", data.get("usage", {})


def run_arm(name, tree, model, port, rep, emit, sizes, out_path, session=None):
    env = dict(
        os.environ,
        PYTHONPATH=os.path.join(tree, "python"),
        YUNSHU_AUTH_DISABLED="1",
        YUNSHU_VLM_APC_DISK="0",
    )
    log = open(f"{os.path.splitext(out_path)[0]}_{name}_{rep}.log", "w")  # noqa: SIM115
    proc = subprocess.Popen(
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
        cwd=tree,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    url = f"http://127.0.0.1:{port}"
    side = None
    try:
        for _ in range(900):
            try:
                urllib.request.urlopen(url + "/v1/models", timeout=2)
                break
            except Exception as exc:  # noqa: BLE001
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"{name}: server exited rc={proc.returncode}"
                    ) from exc
                time.sleep(1)
        else:
            raise RuntimeError(f"{name}: server not ready")
        side = Sidecar(proc.pid, make_fp_reader(proc.pid), lambda: metrics(url))
        side.start()
        step_no = [0]

        def measured(label, messages, max_tokens):
            t0 = time.time()
            out, usage = chat(url, messages, max_tokens)
            t1 = time.time()
            time.sleep(1.5)
            fp, gauges = side.take(t0, t1 + 1.5)
            row = dict(arm=name, rep=rep, step=label, secs=round(t1 - t0, 3))
            row["summary"] = summarize(fp, gauges, t1)
            row["cached"] = (usage.get("prompt_tokens_details") or {}).get(
                "cached_tokens"
            )
            emit(row)
            if label.startswith("96k"):
                emit(
                    dict(
                        arm=name,
                        rep=rep,
                        step=label,
                        trace=[
                            (round(t - t1, 3), round(b / GIB, 3))
                            for t, b in fp[:: max(1, len(fp) // 400)]
                        ],
                        gauges=[(round(t - t1, 3), g) for t, g in gauges],
                    )
                )
            step_no[0] += 1
            return out

        if session:
            import apc_branch_ab

            last = {}

            def chat_fn(msgs, n):
                t0 = time.time()
                out, usage = chat(url, msgs, n)
                t1 = time.time()
                time.sleep(1.5)
                fp, gauges = side.take(t0, t1 + 1.5)
                last["summary"] = summarize(fp, gauges, t1)
                return out, usage, t1 - t0

            def record(step, usage=None, secs=None, ideal=None):
                emit(
                    dict(
                        arm=name,
                        rep=rep,
                        step=step,
                        secs=None if secs is None else round(secs, 3),
                        cached=apc_branch_ab.cached_of(usage) if usage else None,
                        prompt=(usage or {}).get("prompt_tokens"),
                        ideal=ideal,
                        summary=last.get("summary"),
                    )
                )

            apc_branch_ab.scenarios(chat_fn, record, 11 + rep, *session, True)
            return
        for i in range(3):
            measured(
                f"short{i}",
                [{"role": "user", "content": code_doc(100 + i, 900) + "\nSummarize."}],
                256,
            )
        for size in sizes:
            msgs = [
                {
                    "role": "user",
                    "content": code_doc(size, size - 200)
                    + "\nList three functions that use 'cache'.",
                }
            ]
            out = measured(f"{size // 1024}k-turn1", msgs, 128)
            msgs += [
                {"role": "assistant", "content": out},
                {"role": "user", "content": "Now list three that use 'queue'."},
            ]
            measured(f"{size // 1024}k-turn2", msgs, 128)
    finally:
        if side is not None:
            side.close()
        try:
            os.killpg(proc.pid, 2)
            proc.wait(60)
        except Exception:  # noqa: BLE001
            os.killpg(proc.pid, 9)
        log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="name=tree")
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=18996)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--rep-offset", type=int, default=0)
    ap.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--session",
        nargs=3,
        type=int,
        metavar=("TURNS", "PER_TURN", "SUB_TOKENS"),
        help="run the apc_branch_ab long session (two branches) instead of the size ladder",
    )
    a = ap.parse_args()
    arms = [x.split("=", 1) for x in a.arm]
    with open(a.out, "a") as f:

        def emit(row):
            f.write(json.dumps(row) + "\n")
            f.flush()
            if "trace" not in row:
                print(json.dumps(row), flush=True)

        for i in range(a.reps):
            rep = a.rep_offset + i
            order = arms if rep % 2 == 0 else arms[::-1]
            for name, tree in order:
                run_arm(
                    name,
                    tree,
                    a.model,
                    a.port,
                    rep,
                    emit,
                    tuple(a.sizes),
                    a.out,
                    a.session,
                )
        emit(dict(complete=True))


if __name__ == "__main__":
    main()

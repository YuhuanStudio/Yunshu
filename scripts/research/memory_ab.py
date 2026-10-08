"""Server memory A/B: the same 27B checkpoint and request sequence on two source trees.

Each arm starts a fresh server from its tree (PYTHONPATH=<tree>/python, shared .venv),
runs a fixed sequence (short, 8K / 32K / 96K turn 1 + turn 2), and records after every
request the process-tree physical footprint, the sampled peak footprint, APC RAM, the
MLX allocator pool and MLX active / peak memory. Arms alternate (A B A B ...).

    python scripts/research/memory_ab.py --arm v013=/path/tree --arm main=/path/tree \
        --model /path/model --reps 2 --out out.jsonl
"""

import argparse
import contextlib
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from process_memory import process_tree_memory  # noqa: E402

GIB = 2**30
OUT = "memory_ab.jsonl"
SIZES = (8192, 32768, 98304)
WORDS = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "value",
    "index",
    "buffer",
    "layer",
    "cache",
    "token",
    "stream",
    "route",
    "parse",
    "merge",
    "yield",
    "table",
    "schema",
    "record",
    "config",
    "handler",
    "worker",
    "queue",
    "limit",
]


def code_doc(seed, approx_tokens):
    rnd = random.Random(seed)
    lines, n = [], 0
    while n < approx_tokens:
        a, b, c = rnd.sample(WORDS, 3)
        line = f"def {a}_{b}_{n}(x, {c}=None):\n    return x.{c}({rnd.randint(0, 9999)}) + {a}_{n % 97}\n"
        lines.append(line)
        n += 24
    return "".join(lines)


def metrics(url):
    out = {}
    try:
        text = urllib.request.urlopen(url + "/metrics", timeout=10).read().decode()
    except Exception:  # noqa: BLE001
        return out
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        key, val = line.rsplit(" ", 1)
        if (
            key.startswith("yunshu_gpu_memory_bytes")
            or "apc_resident_bytes" in key
            or key.startswith("yunshu_process_footprint_bytes")
        ):
            with contextlib.suppress(ValueError):
                out[key] = round(float(val) / GIB, 3)
    return out


def server_peak_gib(m):
    """Peak footprint from the server's own 20 ms sampler (None when it is not exported)."""
    v = m.get('yunshu_process_footprint_bytes{type="peak"}')
    return None if v is None else v  # metrics() already reports GiB


def chat(url, messages, max_tokens):
    body = json.dumps(
        dict(model="m", messages=messages, max_tokens=max_tokens, temperature=0)
    ).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions", body, {"Content-Type": "application/json"}
    )
    t = time.time()
    data = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    msg = data["choices"][0]["message"]
    return msg.get("content") or "", data.get("usage", {}), time.time() - t


def _tree_python(tree):
    """A tree with the marker `.yv-own-venv` is served from its own .venv (dependency A/B)."""
    py = os.path.join(tree, ".venv", "bin", "python")
    if os.path.exists(os.path.join(tree, ".yv-own-venv")) and os.path.exists(py):
        return py
    return sys.executable


def log_tail(path, lines=15):
    """Last lines of a server log, for the error of a server that died at startup."""
    try:
        with open(path, errors="replace") as f:
            return " | ".join(f.read().strip().splitlines()[-lines:])
    except OSError as exc:
        return f"(no log: {exc})"


def stop_server(proc):
    """SIGINT, then SIGKILL; a server that already exited is not an error."""
    for sig, wait in ((2, 60), (9, 5)):
        try:
            os.killpg(proc.pid, sig)
            proc.wait(wait)
            return
        except ProcessLookupError:
            return
        except Exception:  # noqa: BLE001
            continue


def run_arm(name, tree, model, port, rep, emit, extra_env=None):
    env = dict(
        os.environ,
        PYTHONPATH=os.path.join(tree, "python"),
        YUNSHU_AUTH_DISABLED="1",
        YUNSHU_DEBUG_ROUTES="1",
        YUNSHU_VLM_APC_DISK="0",
        YUNSHU_FOOTPRINT_SAMPLE_MS="20",
    )
    env.update(extra_env or {})
    log = open(f"{os.path.splitext(OUT)[0]}_{name}_{rep}.log", "w")  # noqa: SIM115
    proc = subprocess.Popen(
        [
            _tree_python(tree),
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
    peak = [0]
    stop = threading.Event()
    timeline = []

    def sampler():
        t0, last = time.time(), 0.0
        while not stop.is_set():
            try:
                fp = process_tree_memory(proc.pid)["physical_footprint_sum_bytes"]
                peak[0] = max(peak[0], fp)
                if time.time() - last >= 3.0:
                    last = time.time()
                    m = metrics(url)
                    timeline.append(
                        dict(
                            t=round(last - t0, 1),
                            footprint_gib=round(fp / GIB, 2),
                            active=m.get('yunshu_gpu_memory_bytes{type="active"}'),
                            cache=m.get('yunshu_gpu_memory_bytes{type="cache"}'),
                            apc=m.get('yunshu_apc_resident_bytes{model_id="default"}'),
                        )
                    )
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.2)

    try:
        for _ in range(900):
            try:
                urllib.request.urlopen(url + "/v1/models", timeout=2)
                break
            except Exception as exc:  # noqa: BLE001
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"{name}: server exited rc={proc.returncode}: "
                        + log_tail(log.name)
                    ) from exc
                time.sleep(1)
        else:
            raise RuntimeError(f"{name}: server not ready")
        threading.Thread(target=sampler, daemon=True).start()

        def census(url, step):
            try:
                kv = json.loads(
                    urllib.request.urlopen(url + "/debug/kv-cache", timeout=60).read()
                )
                emit(dict(arm=name, rep=rep, step=step, kv_cache=kv))
            except Exception as exc:  # noqa: BLE001
                emit(dict(arm=name, rep=rep, step=step, kv_cache_error=str(exc)))
            try:
                text = urllib.request.urlopen(
                    url + "/debug/memory-census?min_mib=64", timeout=300
                ).read()
            except Exception as exc:  # noqa: BLE001
                emit(dict(arm=name, rep=rep, step=step, census_error=str(exc)))
                return
            emit(dict(arm=name, rep=rep, step=step, census=json.loads(text)))

        def record(step, usage=None, secs=None):
            fp = process_tree_memory(proc.pid)["physical_footprint_sum_bytes"]
            tl = timeline[:] if "turn" in step else None
            if tl is not None:
                timeline.clear()
            mm = metrics(url)
            sp = server_peak_gib(mm)
            emit(
                dict(
                    arm=name,
                    rep=rep,
                    step=step,
                    footprint_gib=round(fp / GIB, 3),
                    # old field: the larger of the 0.2 s external sampler and the 20 ms in-server one
                    peak_footprint_gib=round(max(peak[0] / GIB, sp or 0.0), 3),
                    peak_external_gib=round(peak[0] / GIB, 3),
                    peak_server_gib=sp,
                    usage=usage,
                    secs=None if secs is None else round(secs, 3),
                    timeline=tl,
                    **mm,
                )
            )

        record("ready")
        for i in range(3):
            _, u, s = chat(
                url,
                [
                    {
                        "role": "user",
                        "content": code_doc(100 + i, 900)
                        + "\nSummarize what this module does.",
                    }
                ],
                256,
            )
            record(f"short{i}", u, s)
        for size in SIZES:
            doc = code_doc(size, size - 200)
            msgs = [
                {
                    "role": "user",
                    "content": doc + "\nList three functions that use 'cache'.",
                }
            ]
            out, u, s = chat(url, msgs, 128)
            record(f"{size // 1024}k-turn1", u, s)
            msgs += [
                {"role": "assistant", "content": out},
                {"role": "user", "content": "Now list three that use 'queue'."},
            ]
            _, u, s = chat(url, msgs, 128)
            record(f"{size // 1024}k-turn2", u, s)
        time.sleep(20)
        record("idle20s")
        time.sleep(15)
        record("idle35s")
        census(url, "idle35s")
        # Does memory held after the long turn return once a short request runs?
        _, u, s = chat(
            url,
            [{"role": "user", "content": code_doc(999, 900) + "\nSummarize this."}],
            64,
        )
        record("short-after", u, s)
        time.sleep(20)
        record("idle-after")
        census(url, "idle-after")
    finally:
        stop.set()
        stop_server(proc)
        log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="name=tree")
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--port", type=int, default=18993
    )  # 18997-18999 are the M3 forwards
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--sizes", type=int, nargs="+", default=[8192, 32768, 98304])
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--arm-env",
        action="append",
        default=[],
        help="name:K=V (server env of that arm)",
    )
    ap.add_argument(
        "--rep-offset", type=int, default=0, help="first rep index (one rep per job)"
    )
    a = ap.parse_args()
    global OUT, SIZES
    OUT, SIZES = a.out, tuple(a.sizes)
    arms = [x.split("=", 1) for x in a.arm]
    complete = 0
    with open(a.out, "a") as f:

        def emit(row):
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(json.dumps(row), flush=True)

        arm_env = {}
        for item in a.arm_env:
            who, kv = item.split(":", 1)
            k, v = kv.split("=", 1)
            arm_env.setdefault(who, {})[k] = v
        for i in range(a.reps):
            rep = a.rep_offset + i
            order = arms if rep % 2 == 0 else arms[::-1]
            for name, tree in order:
                run_arm(name, tree, a.model, a.port, rep, emit, arm_env.get(name))
                complete += 1
        emit(dict(complete=complete == a.reps * len(arms)))


if __name__ == "__main__":
    main()

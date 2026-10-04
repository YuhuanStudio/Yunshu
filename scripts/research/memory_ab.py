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
        if key.startswith("yunshu_gpu_memory_bytes") or "apc_resident_bytes" in key:
            with contextlib.suppress(ValueError):
                out[key] = round(float(val) / GIB, 3)
    return out


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


def run_arm(name, tree, model, port, rep, emit):
    env = dict(
        os.environ,
        PYTHONPATH=os.path.join(tree, "python"),
        YUNSHU_AUTH_DISABLED="1",
        YUNSHU_DEBUG_ROUTES="1",
    )
    log = open(f"{os.path.splitext(OUT)[0]}_{name}_{rep}.log", "w")  # noqa: SIM115
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
    peak = [0]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                fp = process_tree_memory(proc.pid)["physical_footprint_sum_bytes"]
                peak[0] = max(peak[0], fp)
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
                        f"{name}: server exited rc={proc.returncode}"
                    ) from exc
                time.sleep(1)
        else:
            raise RuntimeError(f"{name}: server not ready")
        threading.Thread(target=sampler, daemon=True).start()

        def census(url, step):
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
            emit(
                dict(
                    arm=name,
                    rep=rep,
                    step=step,
                    footprint_gib=round(fp / GIB, 3),
                    peak_footprint_gib=round(peak[0] / GIB, 3),
                    usage=usage,
                    secs=None if secs is None else round(secs, 3),
                    **metrics(url),
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
        census(url, "idle20s")
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
    ap.add_argument("--port", type=int, default=18997)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--sizes", type=int, nargs="+", default=[8192, 32768, 98304])
    ap.add_argument("--out", required=True)
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

        for rep in range(a.reps):
            order = arms if rep % 2 == 0 else arms[::-1]
            for name, tree in order:
                run_arm(name, tree, a.model, a.port, rep, emit)
                complete += 1
        emit(dict(complete=complete == a.reps * len(arms)))


if __name__ == "__main__":
    main()

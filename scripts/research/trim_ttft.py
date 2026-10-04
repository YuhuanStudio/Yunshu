"""TTFT of a prefix-cache hit right after a request vs after the runner went idle.

The idle schedule releases MLX's freed-buffer pool 30 s after the last request; this
measures what the next cache hit pays for it. Each arm starts a fresh server from its
tree; per rep it primes one long document, then times ``max_tokens=1`` requests that hit
the cached document with a new question: immediately (pool warm), after ``--idle``
seconds (pool released on the idle arm) and immediately again.

    python scripts/research/trim_ttft.py --arm main=/tree --arm idle=/tree2 \
        --model /path/model --size 32768 --reps 3 --out out.jsonl
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from memory_ab import chat, code_doc  # noqa: E402


def summarize(rows):
    """Median seconds per phase over rows of {phase, secs}."""
    by = {}
    for r in rows:
        by.setdefault(r["phase"], []).append(r["secs"])
    out = {}
    for phase, vals in by.items():
        vals = sorted(vals)
        out[phase] = vals[len(vals) // 2]
    return out


def run(name, tree, model, port, rep, size, idle, emit):
    env = dict(
        os.environ, PYTHONPATH=os.path.join(tree, "python"), YUNSHU_AUTH_DISABLED="1"
    )
    log = open(f"{os.path.splitext(emit.path)[0]}_{name}_{rep}.log", "w")  # noqa: SIM115
    proc = subprocess.Popen(
        [sys.executable, "-m", "yunshu_cli", "serve", "--model", model]
        + ["--port", str(port)],
        cwd=tree,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    url = f"http://127.0.0.1:{port}"
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
        doc = code_doc(size, size - 200)

        def ask(question):
            m = [{"role": "user", "content": doc + "\n" + question}]
            _, usage, secs = chat(url, m, 1)
            return secs, usage["prompt_tokens_details"]["cached_tokens"]

        ask("List three functions that use 'cache'.")  # prime (cold)
        for phase, wait in (("hit_now", 0), ("hit_after_idle", idle), ("hit_again", 0)):
            time.sleep(wait)
            secs, cached = ask(f"Question {phase} {rep}: name one function.")
            if cached < size // 2:
                raise RuntimeError(f"{name}/{phase}: not a cache hit ({cached})")
            emit(
                dict(arm=name, rep=rep, phase=phase, secs=round(secs, 3), cached=cached)
            )
    finally:
        try:
            os.killpg(proc.pid, 9)
            proc.wait(60)
        except Exception:  # noqa: BLE001
            pass
        log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="name=tree")
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=18996)
    ap.add_argument("--size", type=int, default=32768)
    ap.add_argument("--idle", type=float, default=40.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = [x.split("=", 1) for x in a.arm]
    rows = []
    with open(a.out, "a") as f:

        def emit(row):
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(json.dumps(row), flush=True)

        emit.path = a.out
        for rep in range(a.reps):
            for name, tree in arms if rep % 2 == 0 else arms[::-1]:
                run(name, tree, a.model, a.port, rep, a.size, a.idle, emit)
        for name, _ in arms:
            emit(
                dict(
                    arm=name,
                    median=summarize([r for r in rows if r.get("arm") == name]),
                )
            )
        emit(dict(complete=True))


if __name__ == "__main__":
    main()

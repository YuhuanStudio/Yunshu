"""CPU simulation of admission ordering over captured opencode traces.

Replays three captured sessions (prompt sizes from the request bodies, 4.15
chars/token calibrated on run.jsonl) as concurrent agents against one
single-GPU-thread server model: prefill atoms of 2048 tokens at the measured
236 ms + 1.21 ms/token, batched decode steps of 25 ms. Policies:
  fifo   - upstream: oldest pending prefill first, one decode step per atom
  aux    - auxiliary (tool-less title) requests yield while a main turn waits
  full   - aux + Work.key HRRN uncached ordering + 100 ms decode quanta, aging and bounded overtaking
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from yunshu_engine.serving.work_scheduler import (  # noqa: E402
    DECODE_QUANTUM_S,
    FIXED_S,
    PREFILL_TOKEN_S,
    Work,
)

ATOM = 2048
QUANTUM = [DECODE_QUANTUM_S]
STEP_S = 0.025
CPT = 4.15


def load(runs: Path) -> list[list[tuple[int, bool, int]]]:
    sessions = []
    for d in sorted(
        glob.glob(str(runs / "2026-10-02-origi45/cap-opencode/artifacts/*/bodies"))
    ):
        turns = []
        for f in sorted(glob.glob(d + "/*-req.json")):
            b = json.loads(Path(f).read_text())
            toks = int(len(json.dumps(b)) / CPT)
            turns.append((toks, "tools" not in b, int(b.get("max_tokens") or 0)))
        sessions.append(turns)
    return sessions


class Req:
    def __init__(self, sid, arrive, tokens, cached, aux, gen):
        self.sid, self.arrive, self.aux, self.gen = sid, arrive, aux, gen
        self.left = max(1, tokens - cached)
        self.last = arrive
        self.skips = 0
        self.first = None
        self.dec = 0


def simulate(sessions, policy, starts, think=2.0, gen_main=150, gen_aux=20):
    t = 0.0
    pend, run, done, future = [], [], [], []
    # per-session sequential state: main turns arrive after previous finishes
    nxt = {}
    for sid, turns in enumerate(sessions):
        title = [x for x in turns if x[1]]
        mains = [x for x in turns if not x[1]]
        nxt[sid] = (mains, 0, 0)
        for tok, _, _ in title:
            future.append(Req(sid, starts[sid], tok, 0, True, gen_aux))
        future.append(Req(sid, starts[sid], mains[0][0], 0, False, gen_main))
    debt = 0.0
    mains_ct = {s: 1 for s in nxt}
    while future or pend or run:
        for r in [r for r in future if r.arrive <= t]:
            future.remove(r)
            pend.append(r)
        if not pend and not run:
            t = min(r.arrive for r in future)
            continue
        waiting_main = any(not r.aux for r in pend)

        def key(r):
            if policy == "fifo":
                return (r.arrive,)
            if policy == "aux":
                aged = t - r.last >= 10
                return (0 if (not r.aux or aged or not waiting_main) else 1, r.arrive)
            w = Work(r.arrive, r.last, r.left, -1 if r.aux else 0, skips=r.skips)
            return w.key(t)

        do_decode = bool(run) and (policy != "full" or debt > 0 or not pend)
        if policy == "fifo":
            do_decode = False
        if pend and not (policy == "full" and do_decode):
            r = min(pend, key=key)
            if policy == "full":
                for waiter in pend:
                    if waiter is not r and waiter.aux == r.aux:
                        waiter.skips += 1
            n = min(ATOM, r.left)
            dt = PREFILL_TOKEN_S * n + (0.0 if hasattr(r, "started") else FIXED_S)
            r.started = True
            t += dt
            r.left -= n
            r.last = t
            if r.left <= 0:
                pend.remove(r)
                r.first = t
                run.append(r)
            if policy == "full":
                debt = QUANTUM[0]
            elif run:  # upstream: one decode step per atom
                t += STEP_S
                for x in run[:]:
                    x.dec += 1
        else:
            t += STEP_S
            debt = max(0.0, debt - STEP_S)
            for x in run:
                x.dec += 1
        for x in run[:]:
            if x.dec >= x.gen:
                run.remove(x)
                done.append(x)
                if not x.aux:
                    mains, _, _ = nxt[x.sid]
                    i = mains_ct[x.sid]
                    if i < len(mains):
                        mains_ct[x.sid] += 1
                        tok = mains[i][0]
                        prev = mains[i - 1][0]
                        future.append(
                            Req(x.sid, t + think, tok, min(prev, tok), False, gen_main)
                        )
    return [r.first - r.arrive for r in done if not r.aux], [
        r.first - r.arrive for r in done if r.aux
    ]


def main() -> None:
    global ATOM
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="docs/research/runs")
    ap.add_argument("--atom", type=int, default=ATOM)
    ap.add_argument("--quantum", type=float, default=DECODE_QUANTUM_S)
    ap.add_argument("--offsets", default="0,1.5,3")
    a = ap.parse_args()
    ATOM = a.atom
    if ATOM <= 0 or a.quantum <= 0:
        ap.error("atom and quantum must be positive")
    QUANTUM[0] = a.quantum
    sessions = load(Path(a.runs))
    starts = [float(x) for x in a.offsets.split(",")][: len(sessions)]
    out = {}
    for pol in ("fifo", "aux", "full"):
        m, x = simulate(sessions, pol, starts)
        out[pol] = dict(
            n=len(m),
            main_p50=float(np.percentile(m, 50)),
            main_p90=float(np.percentile(m, 90)),
            main_max=float(max(m)),
            aux_p50=float(np.percentile(x, 50)),
            aux_max=float(max(x)),
        )
        print(pol, {k: round(v, 3) for k, v in out[pol].items()})
    print(json.dumps(out))


if __name__ == "__main__":
    main()

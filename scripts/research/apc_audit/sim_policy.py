"""Offline model of the APC checkpoint store: try policies on a workload before spending GPU.

The model follows mlx_vlm's exact-checkpoint rules for a hybrid (GDN) model: a hit needs a
stored checkpoint whose token list is a prefix of the prompt; every request stores a final
checkpoint (len-1) and interval checkpoints; entries are full copies (bytes = state +
per-token KV) and are evicted LRU by count and by the byte budget.

    sim_policy.py --model M --template body.json --sessions 3 --target 30000 --step 3000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agentic"))

MIB = 1 << 20
GIB = 1 << 30
STATE_BYTES = 160 * MIB  # recurrent (GDN) state per checkpoint, 27B
KV_BYTES_PER_TOKEN = 130 << 10  # measured resident size per token, 27B


@dataclass
class Policy:
    name: str
    entries: int = 2
    interval: int = 2048
    budget_gib: float = 8.0
    head: bool = False
    supersede: bool = False
    disk: bool = False
    one_interval: bool = (
        False  # Yunshu policy: final + 1 interval (+ head), not entries-2 intervals
    )


class Store:
    def __init__(self, p: Policy, head_len=None):
        self.p = p
        self.ram: OrderedDict[tuple, list[int]] = OrderedDict()
        self.disk: dict[tuple, list[int]] = {}
        self.head_marks: set[tuple] = set()
        self.evictions = 0
        self.disk_bytes_written = 0
        self.head_len = head_len

    @staticmethod
    def size(n: int) -> int:
        return STATE_BYTES + n * KV_BYTES_PER_TOKEN

    def resident(self) -> int:
        return sum(self.size(len(t)) for t in self.ram.values())

    def lookup(self, ids: list[int]) -> tuple[int, str]:
        best, key, tier = 0, None, "none"
        for src, name in ((self.ram, "ram"), (self.disk, "ssd")):
            for k, stored in src.items():
                n = len(stored)
                if best < n < len(ids) and ids[:n] == stored:
                    best, key, tier = n, k, name
        if key is not None and tier == "ram":
            self.ram.move_to_end(key)
        elif key is not None:
            self._put(key, self.disk[key])  # promote
        return best, tier

    def _spill(self, k, toks):
        if self.p.disk and k not in self.disk:
            self.disk[k] = toks
            self.disk_bytes_written += self.size(len(toks))

    def _put(self, k, toks):
        budget = int(self.p.budget_gib * GIB)
        sz = self.size(len(toks))
        if sz > budget:
            self._spill(k, toks)
            return
        self.ram[k] = toks
        self.ram.move_to_end(k)
        while len(self.ram) > self.p.entries or self.resident() > budget:
            ek, et = self.ram.popitem(last=False)
            self.evictions += 1
            self._spill(ek, et)

    def store(self, ids: list[int], prefix: int) -> None:
        final = len(ids) - 1
        lengths = {final}
        p = self.p
        if p.entries > 1 and p.interval > 0:
            last = ((final - 1) // p.interval) * p.interval
            first = max(
                p.interval,
                last - (0 if p.one_interval else (p.entries - 2)) * p.interval,
            )
            for b in range(first, last + 1, p.interval):
                if 16 <= b < final:
                    lengths.add(b)
        if p.head and self.head_len:
            h = self.head_len(ids)
            if h and 16 <= h < final:
                lengths.add(h)
        for n in sorted(lengths):
            if n <= prefix:
                continue
            k = tuple(ids[:n])
            if p.supersede and n == final:
                # drop earlier checkpoints of this lineage (except the head)
                for ek in [
                    ek
                    for ek in self.ram
                    if len(ek) < n
                    and ids[: len(ek)] == list(ek)
                    and ek not in self.head_marks
                ]:
                    self.ram.pop(ek)
            self._put(k, ids[:n])
            if p.head and self.head_len and n == self.head_len(ids):
                self.head_marks.add(k)


def run(policy: Policy, workload: list[list[int]], head_len=None):
    st = Store(policy, head_len)
    rows = []
    for ids in workload:
        hit, tier = st.lookup(ids)
        st.store(ids, hit)
        rows.append((len(ids), hit, tier))
    tot = sum(r[0] for r in rows)
    return dict(
        policy=policy.name,
        cached_ratio=round(sum(r[1] for r in rows) / tot, 3),
        reprefill_tokens=tot - sum(r[1] for r in rows),
        ram_hits=sum(1 for r in rows if r[2] == "ram"),
        ssd_hits=sum(1 for r in rows if r[2] == "ssd"),
        misses=sum(1 for r in rows if r[2] == "none"),
        evictions=st.evictions,
        disk_written_gib=round(st.disk_bytes_written / GIB, 1),
        rows=rows,
    )


def build_workload(a, tok, render):
    from session_replay import Session

    template = json.loads(Path(a.template).read_text())
    sess = [Session(i, template, tok, a.step, a.target) for i in range(a.sessions)]
    steps = max(1, (a.target - 7500) // a.step)
    order = []
    for k in range(1, steps + 1):
        for s in sess:
            order.append((s, k))
    for s in sess:
        order.append((s, steps))
    for s in sess:
        order.append((s, steps + 1))
    return [render(s.body(k))["ids"] for s, k in order], [(s.idx, k) for s, k in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--target", type=int, default=30000)
    ap.add_argument("--step", type=int, default=3000)
    a = ap.parse_args()
    logging.disable(logging.CRITICAL)
    from render_bodies import Renderer

    r = Renderer(a.model)
    tok = r.engine._tokenizer
    ids, labels = build_workload(a, tok, r.render)
    ustart = tok.convert_tokens_to_ids("<|im_start|>")
    user_tok = tok.encode("user", add_special_tokens=False)[0]

    def head_len(x):
        for i in range(len(x) - 1):
            if x[i] == ustart and x[i + 1] == user_tok:
                return i
        return 0

    print(
        f"workload: {len(ids)} requests, prompts {ids and len(ids[0])}..{max(map(len, ids))}"
    )
    y = dict(entries=8, one_interval=True, supersede=True, head=True)
    policies = [
        Policy("baseline (2 entries, 8 GiB)"),
        Policy("baseline + SSD (write-through)", disk=True),
        Policy("yunshu 8 GiB", **y),
        Policy("yunshu 8 GiB + SSD", disk=True, **y),
        Policy("yunshu 16 GiB", budget_gib=16, **y),
        Policy("yunshu 32 GiB", budget_gib=32, **y),
        Policy("yunshu 32 GiB, no interval", budget_gib=32, **{**y, "interval": 0}),
        Policy("baseline 32 GiB, 2 entries", budget_gib=32),
        Policy("baseline 32 GiB, 8 entries", budget_gib=32, entries=8),
    ]
    for p in policies:
        res = run(p, ids, head_len)
        rows = res.pop("rows")
        print(json.dumps(res))
        if "-v" in sys.argv:
            for lab, (n, h, t) in zip(labels, rows, strict=True):
                print("   ", lab, n, h, t)


if __name__ == "__main__":
    main()

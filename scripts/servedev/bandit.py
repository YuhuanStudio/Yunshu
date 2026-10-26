"""Lossless bandit autotuner: UCB1 over knob settings that must not change output.

An arm whose replay tokens differ from the baseline is banned for good (the knob was not lossless after all;
e.g. prefill chunk size is documented as not bit-identical). Promotion needs margin, sample count and zero
identity failures, and is recorded in an append-only ledger so it can be reverted.
"""

from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path

DEFAULT_SPACE = {
    # Only registered knobs.
    # Add YUNSHU_GC_POLICY once the ttftfloor line lands it in the registry.
    "YUNSHU_MTP_BLOCK_SIZE": ["4", "6", "8"],
}


def arms_from(space: dict[str, list[str]]) -> list[tuple]:
    keys = sorted(space)
    return [
        tuple(zip(keys, vals, strict=True))
        for vals in itertools.product(*(space[k] for k in keys))
    ]


class Bandit:
    def __init__(
        self, arms: list[tuple], baseline: tuple, min_n: int = 5, margin: float = 0.03
    ):
        if baseline not in arms:
            raise ValueError("baseline must be one of the arms")
        self.arms, self.baseline, self.min_n, self.margin = (
            arms,
            baseline,
            min_n,
            margin,
        )
        self.rewards: dict[tuple, list[float]] = {a: [] for a in arms}
        self.banned: dict[tuple, str] = {}
        self.t = 0

    def choose(self) -> tuple | None:
        live = [a for a in self.arms if a not in self.banned]
        for a in live:  # cold start: baseline first, then the rest
            if not self.rewards[a] and a == self.baseline:
                return a
        for a in live:
            if not self.rewards[a]:
                return a
        if not live:
            return None
        self.t += 1
        return max(
            live,
            key=lambda a: (
                self._mean(a)
                + math.sqrt(2 * math.log(self.t + 1) / len(self.rewards[a]))
            ),
        )

    def _mean(self, a: tuple) -> float:
        r = self.rewards[a]
        return sum(r) / len(r)

    def observe(self, arm: tuple, reward: float, identical: bool = True) -> None:
        """reward: higher is better (e.g. decode_tps / base_decode_tps, or 1/ttft ratio). Not identical => banned."""
        if arm == self.baseline:
            identical = True  # the baseline defines the reference tokens
        if not identical:
            self.banned[arm] = "tokens differ from baseline"
            return
        self.rewards[arm].append(reward)

    def best(self) -> tuple | None:
        """A non-baseline arm that beats the baseline by margin with enough samples, else None."""
        base = self.rewards[self.baseline]
        if len(base) < self.min_n:
            return None
        bm = self._mean(self.baseline)
        pool = [
            a
            for a in self.arms
            if a not in self.banned
            and a != self.baseline
            and len(self.rewards[a]) >= self.min_n
        ]
        if not pool:
            return None
        top = max(pool, key=self._mean)
        return top if self._mean(top) >= bm * (1 + self.margin) else None


class Ledger:
    """Append-only promotion log. revert() appends the inverse; nothing is rewritten."""

    def __init__(self, path: Path):
        self.path = path

    def append(self, rec: dict) -> dict:
        rec = {"t": time.time(), **rec}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def read(self) -> list[dict]:
        return (
            [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]
            if self.path.exists()
            else []
        )

    def active(self) -> dict[str, str]:
        """Current promoted value per knob (promote sets, revert restores the recorded previous value)."""
        cur: dict[str, str] = {}
        for r in self.read():
            for k, v in (r.get("set") or {}).items():
                if r["kind"] in ("promote", "revert"):
                    cur[k] = v
        return cur

    def promote(self, new: dict[str, str], old: dict[str, str], evidence: dict) -> dict:
        return self.append(
            {"kind": "promote", "set": new, "previous": old, "evidence": evidence}
        )

    def revert_last(self, reason: str) -> dict:
        ps = [r for r in self.read() if r["kind"] == "promote"]
        reverted = {r["of"] for r in self.read() if r["kind"] == "revert"}
        open_ = [r for r in ps if r["t"] not in reverted]
        if not open_:
            raise LookupError("nothing to revert")
        last = open_[-1]
        return self.append(
            {
                "kind": "revert",
                "of": last["t"],
                "set": last["previous"],
                "reason": reason,
            }
        )

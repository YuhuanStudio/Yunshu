"""Synchronized per-stage timer of the DFlash round (YUNSHU_DEBUG_TREE_PROFILE).

With the switch on, every stage boundary evaluates the stage's arrays and charges the wall time since
the previous boundary to that stage, so the split is exact per stage while the round itself gets
longer than the pipelined real one (a diagnostic, never a speed number). When a request ends, its
per-round means are logged as one ``YUNSHU_LAYER2 {json}`` line, the layer-2 contract the
comparison harness reads (scripts/research/fullcmp_profile.py, fullcmp_attribution.LAYER2_YUNSHU):

* ``drafter``: the DFlash drafter forward and the tree lattice built from it
* ``plan``: the tree search and reorder that choose the verified nodes
* ``copy_lookup``: the copy drafter's n-gram lookup
* ``verify_attention``: tree attention over the KV in the full-attention layers (solo rows; a packed
  multi-row round does not split it and reports only ``verify_total``)
* ``verify_weights``: the rest of the verify forward, logits and target draws (projections, GDN,
  MLP, norms, sampling)
* ``commit``: readback, acceptance walk, emission, KV commit and drafter ring append

Off (the default) every call is a no-op and nothing is evaluated early.
"""

from __future__ import annotations

import json
import logging
import time

import mlx.core as mx

logger = logging.getLogger(__name__)

STAGES = (
    "drafter",
    "plan",
    "copy_lookup",
    "verify_weights",
    "verify_attention",
    "commit",
)


def enabled() -> bool:
    from .. import settings

    return bool(settings.get("YUNSHU_DEBUG_TREE_PROFILE"))


class RoundProfile:
    """Per-row stage sums for one round driver; ``on`` is read once at construction."""

    def __init__(self, on: bool | None = None):
        self.on = enabled() if on is None else on
        self._t = 0.0
        self._round: dict[str, float] = {}
        self._rows: dict[int, dict[str, float]] = {}

    def begin(self) -> None:
        if not self.on:
            return
        self._round = {}
        self._t = time.perf_counter()

    def mark(self, stage: str, *arrays) -> None:
        """Evaluate ``arrays`` and charge the time since the last boundary to ``stage``."""
        if not self.on:
            return
        live = [a for a in arrays if isinstance(a, mx.array)]
        if live:
            mx.eval(*live)
        now = time.perf_counter()
        self._round[stage] = self._round.get(stage, 0.0) + (now - self._t) * 1000
        self._t = now

    def end(self, row_ids, attention_ms: float | None) -> None:
        """Close the round: split verify into attention / weights and add it to every row in it."""
        if not self.on:
            return
        r = self._round
        total = r.pop("verify", 0.0)
        if attention_ms is not None:
            att = min(total, attention_ms)
            r["verify_attention"] = att
            r["verify_weights"] = total - att
        else:
            r["verify_total"] = total
        for rid in row_ids:
            acc = self._rows.setdefault(rid, {"rounds": 0.0})
            acc["rounds"] += 1
            for k, v in r.items():
                acc[k] = acc.get(k, 0.0) + v

    def finish(self, rid: int) -> dict | None:
        """Log and return the request's per-round means (None when off or the row never ran a round)."""
        acc = self._rows.pop(rid, None)
        if not self.on or not acc or not acc.get("rounds"):
            return None
        n = acc.pop("rounds")
        stages = {k: round(v / n, 3) for k, v in sorted(acc.items())}
        payload = {"rounds": int(n), "stages": {"decode_round": stages}}
        logger.info("YUNSHU_LAYER2 %s", json.dumps(payload, sort_keys=True))
        return payload

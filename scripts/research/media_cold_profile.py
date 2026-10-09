"""Where do the extra milliseconds of a non-hit media turn go? (inclusive wall time per stage)

Wraps candidate stages with accumulating timers, drives cold/miss follow-ups through the
real VLMEngine, and writes one JSON row per arm. Run the same file against the base tree
(missing targets are skipped and recorded) for the A/B.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import importlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

TARGETS = (
    ("mlx_vlm.utils", "prepare_inputs", "prepare_inputs"),
    ("mlx_vlm.apc", "hash_image_payload", "media_key.hash_pixels"),
    ("mlx_vlm.apc", "semantic_extra_hash", "media_key.semantic_hash"),
    (
        "yunshu_engine.vlm_batch_runner",
        "VLMBatchRunner.prepare_media",
        "prepare_media(total)",
    ),
    ("yunshu_engine.vlm_batch_runner", "VLMBatchRunner._admit", "admit"),
    (
        "yunshu_engine.apc_manager",
        "YunshuAPCManager.lookup_exact_cache",
        "lookup_exact_cache",
    ),
    (
        "yunshu_engine.apc_manager",
        "YunshuAPCManager.store_exact_cache",
        "store_exact_cache",
    ),
    ("yunshu_engine.apc_manager", "_Coordinator.lookup", "coordinator.lookup"),
    ("yunshu_engine.apc_manager", "_Coordinator.store_checkpoint", "checkpoint.store"),
    (
        "yunshu_engine.apc_manager",
        "_Coordinator.checkpoint_lengths",
        "checkpoint.lengths",
    ),
    (
        "yunshu_engine.apc_manager",
        "_Coordinator.flush_deferred_checkpoints",
        "checkpoint.flush",
    ),
    (
        "yunshu_engine.apc_restore_cost",
        "MediaRestoreCost.worth",
        "restore_skip.cost_check",
    ),
)


class Timers:
    def __init__(self):
        self.totals: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.notes: dict[str, str] = {}

    def wrap(self, label, fn):
        @functools.wraps(fn)
        def inner(*a, **kw):
            if label == "checkpoint.store":
                # why the capture can/cannot be deferred past the first token
                cache = kw.get("prompt_cache", a[2] if len(a) > 2 else ())
                self.notes["checkpoint_cache_types"] = ",".join(
                    sorted({type(c).__name__ for c in cache})
                )
                self.notes["defer_flag"] = str(
                    getattr(a[0], "defer_checkpoint_stores", None)
                )
            t = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                self.totals[label] = (
                    self.totals.get(label, 0.0) + time.perf_counter() - t
                )
                self.calls[label] = self.calls.get(label, 0) + 1

        return inner

    def reset(self):
        self.totals.clear()
        self.calls.clear()

    def snapshot_ms(self):
        return {k: round(v * 1000, 3) for k, v in sorted(self.totals.items())}


def install(timers: Timers, targets=TARGETS):
    """Patch each importable target; return (installed, missing) label lists."""
    done, missing = [], []
    for mod, dotted, label in targets:
        try:
            owner = importlib.import_module(mod)
            *path, name = dotted.split(".")
            for p in path:
                owner = getattr(owner, p)
            setattr(owner, name, timers.wrap(label, getattr(owner, name)))
            done.append(label)
        except (ImportError, AttributeError):
            missing.append(label)
    return done, missing


def summarize(samples: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({k for s in samples for k in s})
    return {
        k: round(statistics.median(s.get(k, 0.0) for s in samples), 3) for k in keys
    }


async def run(a):
    import multimodal_apc as m

    from yunshu_engine.vlm_engine import VLMEngine

    timers = Timers()
    done, missing = install(timers)
    engine = VLMEngine(a.model)
    await engine.start()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    msg = m.messages(a.size, tokenizer=engine._tokenizer)
    timers.reset()
    first = await m.probe(engine, msg, cold=True)
    first_stages = timers.snapshot_ms()
    follow = msg + [
        {"role": "assistant", "content": first["text"]},
        {"role": "user", "content": "What colour did you see? Answer with one word."},
    ]
    modes = {"miss": dict(cold=True), "hit_or_skip": dict(cold=False)}
    rows = {}
    for name, kw in modes.items():
        ttfts, stages = [], []
        for _ in range(a.reps):
            if name == "hit_or_skip":  # restore the first-turn checkpoint
                await m.probe(engine, msg, cold=True)
            timers.reset()
            r = await m.probe(engine, follow, **kw)
            ttfts.append(r["ttft_s"] * 1000)
            stages.append(timers.snapshot_ms())
        rows[name] = dict(
            ttft_ms_median=round(statistics.median(ttfts), 2),
            ttft_ms=[round(x, 2) for x in ttfts],
            stages_ms_median=summarize(stages),
            skipped=r.get("restore_skipped"),
            cached=r["cached"],
        )
    out.write_text(
        json.dumps(
            dict(
                event="media_cold_profile",
                complete=True,
                model=a.model,
                notes=timers.notes,
                first_request=dict(
                    ttft_ms=round(first["ttft_s"] * 1000, 2), stages_ms=first_stages
                ),
                installed=done,
                missing=missing,
                rows=rows,
            ),
            indent=1,
        )
        + "\n"
    )
    print(json.dumps(rows, indent=1), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=46)
    p.add_argument("--reps", type=int, default=12)
    asyncio.run(run(p.parse_args()))

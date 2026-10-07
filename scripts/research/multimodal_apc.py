"""Lossless media-prefix checkpoint probe, run as a pinned yv cell through gpuq.

One model load per arm. Capture the runner's raw token IDs (including hidden
reasoning/control tokens), not re-tokenized output. TTFT includes preprocessing.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import time
from pathlib import Path

from omni_apc_probe import png


def messages(size: int, rgb=(200, 30, 30)) -> list:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,"
                        + base64.b64encode(png(rgb=rgb)).decode()
                    },
                },
                {
                    "type": "text",
                    "text": "The image is part of this conversation. " * size
                    + "Describe its colour in one word.",
                },
            ],
        }
    ]


def validate_pair(cold: dict, warm: dict, require_hit: bool) -> None:
    if not cold["ids"] or cold["ids"] != warm["ids"]:
        raise ValueError("APC hit != miss raw token IDs")
    if require_hit and warm["cached"] <= 0:
        raise ValueError("APC not engaged")
    if cold["cached"] != 0:
        raise ValueError("cold control unexpectedly hit")


async def probe(engine, msg: list, *, cold=False):
    # Flush the RAM manager rather than change kernels/settings between controls.
    if cold and engine._apc_backend is not None:
        engine._apc_backend.clear()
    ids = []
    original = engine._runner_events

    def capture(*args, **kwargs):
        for event in original(*args, **kwargs):
            if event[1] is not None:
                ids.append(int(event[1]))
            yield event

    engine._runner_events = capture
    start = time.perf_counter()
    first = None
    text = []
    last = None
    try:
        async for item in engine.generate_stream(
            messages=msg, temperature=0, max_tokens=32, enable_thinking=False
        ):
            last = item
            if item.new_token_ids and first is None:
                first = time.perf_counter()
            text.append(item.new_text)
    finally:
        engine._runner_events = original
    if last is None or not last.finished or not ids:
        raise ValueError("incomplete generation")
    return dict(
        ids=ids,
        sha=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        cached=last.cached_tokens,
        pt=last.prompt_tokens,
        ttft_s=(first or time.perf_counter()) - start,
        text="".join(text),
    )


async def run(a):
    from yunshu_engine.vlm_engine import VLMEngine

    engine = VLMEngine(a.model)
    await engine.start()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out.open("w") as f:

            def emit(row):
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(
                    json.dumps(
                        {k: v for k, v in row.items() if k not in ("ids", "text")}
                    ),
                    flush=True,
                )

            emit(
                {
                    "event": "engaged",
                    "runner": type(engine._batch_runner).__name__,
                    "model": a.model,
                }
            )
            for size in a.sizes:
                msg = messages(size)
                cold = await probe(engine, msg, cold=True)
                warm = await probe(engine, msg)
                validate_pair(cold, warm, a.require_hit)
                emit({"event": "request", "kind": "cold", "size": size, **cold})
                emit({"event": "request", "kind": "warm", "size": size, **warm})
                follow = msg + [
                    {"role": "assistant", "content": cold["text"]},
                    {
                        "role": "user",
                        "content": "What colour did you see? Answer with one word.",
                    },
                ]
                hit = await probe(engine, follow)
                miss = await probe(engine, follow, cold=True)
                validate_pair(miss, hit, a.require_hit)
                emit({"event": "request", "kind": "turn2-hit", "size": size, **hit})
                emit({"event": "request", "kind": "turn2-miss", "size": size, **miss})
                # Same placeholder IDs, different pixels must never reuse the image state.
                other = await probe(engine, messages(size, (30, 30, 200)))
                if other["cached"]:
                    raise ValueError("foreign image reused pixel state")
                emit({"event": "request", "kind": "other-image", "size": size, **other})
            emit({"complete": True})
    finally:
        await engine.stop()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 4096])
    p.add_argument("--require-hit", action="store_true")
    return p


if __name__ == "__main__":
    asyncio.run(run(parser().parse_args()))

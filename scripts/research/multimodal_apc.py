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


def messages(size: int, rgb=(200, 30, 30), tokenizer=None) -> list:
    question = "Describe its colour in one word."
    text = "The image is part of this conversation. " * size + question
    if tokenizer is not None:
        tail = tokenizer.encode(question, add_special_tokens=False)
        unit = tokenizer.encode(
            "The image is part of this conversation. ", add_special_tokens=False
        )
        n = max(0, size - len(tail))
        tokens = (unit * ((n + len(unit) - 1) // len(unit)))[:n]
        text = tokenizer.decode(tokens + tail)
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
                    "text": text,
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


async def anthropic_probe(engine, client, body, *, cold=False):
    if cold and engine._apc_backend is not None:
        engine._apc_backend.clear()
    ids = []
    original = engine._runner_events
    start = time.perf_counter()
    first = None

    def capture(*args, **kwargs):
        nonlocal first
        for event in original(*args, **kwargs):
            if event[1] is not None:
                ids.append(int(event[1]))
                if first is None:
                    first = time.perf_counter()
            yield event

    engine._runner_events = capture
    try:
        response = await client.post("/v1/messages", json=body)
    finally:
        engine._runner_events = original
    response.raise_for_status()
    data = response.json()
    if data.get("type") != "message" or not data.get("stop_reason") or not ids:
        raise ValueError(f"incomplete Anthropic response: {data}")
    usage = data["usage"]
    return dict(
        ids=ids,
        sha=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        cached=usage.get("cache_read_input_tokens", 0),
        created=usage.get("cache_creation_input_tokens", 0),
        pt=sum(
            usage.get(k, 0)
            for k in (
                "input_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
            )
        ),
        ttft_s=first - start,
        text="".join(b.get("text", "") for b in data["content"]),
    )


def anthropic_body(model):
    return dict(
        model=model,
        max_tokens=32,
        temperature=0,
        thinking={"type": "disabled"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.b64encode(png()).decode(),
                        },
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": "Describe the image's colour in one word.",
                    },
                ],
            }
        ],
    )


async def run(a):
    import httpx

    from yunshu_engine.vlm_engine import VLMEngine
    from yunshu_gateway.engine import set_engine
    from yunshu_gateway.main import create_app

    engine = VLMEngine(a.model)
    await engine.start()
    set_engine(engine)
    app = create_app()
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
                msg = messages(size, tokenizer=engine._tokenizer)
                cold = await probe(engine, msg, cold=True)
                warm = await probe(engine, msg)
                emit({"event": "request", "kind": "cold", "size": size, **cold})
                emit({"event": "request", "kind": "warm", "size": size, **warm})
                validate_pair(cold, warm, a.require_hit)
                follow = msg + [
                    {"role": "assistant", "content": cold["text"]},
                    {
                        "role": "user",
                        "content": "What colour did you see? Answer with one word.",
                    },
                ]
                hit = await probe(engine, follow)
                miss = await probe(engine, follow, cold=True)
                emit({"event": "request", "kind": "turn2-hit", "size": size, **hit})
                emit({"event": "request", "kind": "turn2-miss", "size": size, **miss})
                validate_pair(miss, hit, a.require_hit)
                # Same placeholder IDs, different pixels must never reuse the image state.
                other = await probe(
                    engine, messages(size, (30, 30, 200), tokenizer=engine._tokenizer)
                )
                if other["cached"]:
                    raise ValueError("foreign image reused pixel state")
                emit({"event": "request", "kind": "other-image", "size": size, **other})
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://yunshu"
            ) as client:
                body = anthropic_body(engine.model_name)
                cold = await anthropic_probe(engine, client, body, cold=True)
                warm = await anthropic_probe(engine, client, body)
                emit({"event": "request", "kind": "anthropic-cold", "size": 0, **cold})
                emit({"event": "request", "kind": "anthropic-warm", "size": 0, **warm})
                validate_pair(cold, warm, a.require_hit)
                if a.require_hit and not cold["created"]:
                    raise ValueError("image cache_control did not create a checkpoint")
                body["messages"] += [
                    {"role": "assistant", "content": cold["text"]},
                    {"role": "user", "content": "What colour was the image? One word."},
                ]
                hit = await anthropic_probe(engine, client, body)
                miss = await anthropic_probe(engine, client, body, cold=True)
                emit(
                    {
                        "event": "request",
                        "kind": "anthropic-turn2-hit",
                        "size": 0,
                        **hit,
                    }
                )
                emit(
                    {
                        "event": "request",
                        "kind": "anthropic-turn2-miss",
                        "size": 0,
                        **miss,
                    }
                )
                validate_pair(miss, hit, a.require_hit)
            emit({"complete": True})
    finally:
        set_engine(None)
        await engine.stop()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 32768])
    p.add_argument("--require-hit", action="store_true")
    return p


if __name__ == "__main__":
    asyncio.run(run(parser().parse_args()))

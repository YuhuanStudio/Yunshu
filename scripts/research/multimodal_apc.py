"""Lossless media-prefix checkpoint probe, run as a pinned yv cell through gpuq.

One model load per arm. Capture the runner's raw token IDs (including hidden
reasoning/control tokens), not re-tokenized output. TTFT includes preprocessing.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import hashlib
import io
import json
import math
import os
import struct
import sys
import time
import wave
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev"))
from gpuq_pause import was_paused  # noqa: E402
from omni_apc_probe import png  # noqa: E402
from prompt_len import check_served_length  # noqa: E402


def with_audio(msg):
    """A deterministic PCM fixture exercises joint image/audio feature keys."""
    audio = io.BytesIO()
    with wave.open(audio, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(
            b"".join(
                struct.pack("<h", int(2000 * math.sin(2 * math.pi * 440 * i / 16000)))
                for i in range(8000)
            )
        )
    msg = copy.deepcopy(msg)
    msg[0]["content"].insert(
        1,
        dict(
            type="input_audio",
            input_audio=dict(
                data=base64.b64encode(audio.getvalue()).decode(), format="wav"
            ),
        ),
    )
    return msg


def messages(size: int, rgb=(200, 30, 30), overhead=None, tokenizer=None) -> list:
    """Image + filler text whose SERVED length (usage.prompt_tokens, turn 1) is `size`.

    overhead = served tokens that are not text (chat template + image [+ audio]); measure it with
    `measure_overhead`. size <= overhead + 8 (or no tokenizer) gives the minimal one-phrase request, whose label
    is whatever the server reports (see prompt_len.py for the length-label definition)."""
    question = "Describe its colour in one word."
    text = "The image is part of this conversation. " + question
    if tokenizer is not None and overhead is not None and size > overhead + 8:

        def count(t):
            return len(tokenizer.encode(t, add_special_tokens=False))

        source = "The image is part of this conversation. " * (size // 4 + 8)
        ids = tokenizer.encode(source, add_special_tokens=False)
        tail = "\n\n" + question
        want = size - overhead
        n = max(1, want - count(tail))
        for _ in range(32):
            text = tokenizer.decode(ids[:n]) + tail
            actual = count(text)
            if actual == want:
                break
            n += want - actual
        else:
            raise ValueError(
                f"cannot construct served length {size} (text {actual} != {want})"
            )
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


def text_tokens(msg, tokenizer):
    parts = [
        c["text"] for c in msg[0]["content"] if isinstance(c, dict) and "text" in c
    ]
    return sum(len(tokenizer.encode(t, add_special_tokens=False)) for t in parts)


async def measure_overhead(engine, with_audio_too=False):
    """Served tokens that are not filler text: template + image (+ audio), from one minimal request."""
    msg = messages(1)
    if with_audio_too:
        msg = with_audio(msg)
    r = await probe(engine, msg, cold=True)
    return r["pt"] - text_tokens(msg, engine._tokenizer)


def validate_pair(
    cold: dict, warm: dict, require_hit: bool, *, allow_skip=False
) -> None:
    if not cold["ids"] or cold["ids"] != warm["ids"]:
        raise ValueError("APC hit != miss raw token IDs")
    if (
        require_hit
        and warm["cached"] <= 0
        and not (allow_skip and warm.get("restore_skipped") is True)
    ):
        raise ValueError("APC not engaged")
    if cold["cached"] != 0:
        raise ValueError("cold control unexpectedly hit")


async def probe(engine, msg: list, *, cold=False):
    # Flush the RAM manager rather than change kernels/settings between controls.
    if cold and engine._apc_backend is not None:
        engine._apc_backend.clear()
    cost = getattr(engine._apc_backend, "media_restore_cost", None)
    skips = getattr(cost, "skips", 0)
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
        restore_skipped=getattr(cost, "skips", 0) > skips,
        pt=last.prompt_tokens,
        ttft_s=(first or time.perf_counter()) - start,
        text="".join(text),
    )


async def measured(call, restore=None):
    """Discard paused timing samples, preserving the original prefix on retry."""
    if os.getenv("GPUQ_DEVICE") == "m3":
        result = await call()
        result["timing_valid"] = False  # portability evidence, never a timing decision
        return result
    for attempt in range(5):
        if attempt and restore is not None:
            await restore()
        start = time.time()
        result = await call()
        if not was_paused(start, time.time()):
            result.update(timing_valid=True, paused_retries=attempt)
            return result
    raise RuntimeError("TTFT repeatedly overlapped gpuq pauses")


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


ANTHROPIC_TEXT = (
    "You are checking a test image for a colour calibration report. The report lists "
    "the dominant colour of every swatch, one word each, using plain colour names such "
    "as red, green, blue, yellow, black or white, and never a shade or a hex code. "
    "Look at the whole image, ignore any border, and decide which single colour covers "
    "most of it. Describe the image's colour in one word."
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
                    },
                    {
                        # The breakpoint sits on a text block long enough that a 64-aligned
                        # checkpoint lands after the image: checkpoints never cut a media
                        # span, so a one-line question after a ~60-token image (27B) could
                        # never be cached and the warm request could never hit.
                        "type": "text",
                        "text": ANTHROPIC_TEXT,
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            }
        ],
    )


def arithmetic_item(i):
    """A deterministic image-conditioned paired set, with known answers."""
    a, b = 100 + (i * 17) % 89, 1 + (i * 23) % 97
    red = i % 2 == 0
    msg = messages(1, (200, 30, 30) if red else (30, 30, 200))
    # The fixed instructions keep the text after the image long enough for a 64-aligned
    # checkpoint to land past the image span (checkpoints never cut one): without them a
    # ~114-token item on 27B had no legal checkpoint and the hit request could never hit.
    msg[0]["content"][1]["text"] = (
        "You are checking a colour-coded arithmetic card for a calibration report. "
        "The card is a single solid swatch. Read its colour from the whole image, "
        "ignore any border, then follow the rule for that colour exactly and answer "
        "with digits only, no words, no punctuation and no explanation. "
        f"If the image is red, compute {a} + {b}. If it is blue, compute {a} - {b}. "
        "Reply with only the integer answer."
    )
    return msg, a + b if red else a - b


async def run(a):
    import httpx

    from yunshu_engine import settings
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
                    "image_audio": a.image_audio,
                    "model": a.model,
                    "settings": {
                        k: settings.get(k)
                        for k in (
                            "YUNSHU_KV_PRECISION",
                            "YUNSHU_VLM_APC_WARM",
                            "YUNSHU_VLM_DRAFT",
                        )
                    },
                    "versions": {
                        k: version(k)
                        for k in ("mlx", "mlx-lm", "mlx-vlm", "transformers")
                    },
                }
            )
            overhead = await measure_overhead(engine, a.image_audio)
            emit({"event": "overhead", "non_text_tokens": overhead})
            tk = engine._tokenizer
            for size in a.sizes:
                msg = messages(size, overhead=overhead, tokenizer=tk)
                if a.image_audio:
                    msg = with_audio(msg)
                cold = await measured(lambda: probe(engine, msg, cold=True))
                if size > overhead + 8:
                    # label = served turn-1 length (prompt_len.py); fail the cell on mismatch
                    check_served_length(size, {"prompt_tokens": cold["pt"]})
                warm = await measured(
                    lambda: probe(engine, msg),
                    restore=lambda: probe(engine, msg, cold=True),
                )
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
                hit = await measured(
                    lambda: probe(engine, follow),
                    restore=lambda: probe(engine, msg, cold=True),
                )
                miss = await measured(lambda: probe(engine, follow, cold=True))
                emit({"event": "request", "kind": "turn2-hit", "size": size, **hit})
                emit({"event": "request", "kind": "turn2-miss", "size": size, **miss})
                validate_pair(miss, hit, a.require_hit, allow_skip=True)
                # Same placeholder IDs, different pixels must never reuse the image state.
                other = await measured(
                    lambda: probe(
                        engine,
                        (
                            with_audio(
                                messages(
                                    size, (30, 30, 200), overhead=overhead, tokenizer=tk
                                )
                            )
                            if a.image_audio
                            else messages(
                                size, (30, 30, 200), overhead=overhead, tokenizer=tk
                            )
                        ),
                    ),
                    restore=lambda: probe(engine, follow, cold=True),
                )
                if other["cached"]:
                    raise ValueError("foreign image reused pixel state")
                emit({"event": "request", "kind": "other-image", "size": size, **other})
            if a.skip_anthropic:
                emit(
                    {"event": "scope", "anthropic": "candidate-only cold/hit controls"}
                )
            else:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://yunshu"
                ) as client:
                    body = anthropic_body(engine.model_name)
                    cold = await measured(
                        lambda: anthropic_probe(engine, client, body, cold=True)
                    )
                    warm = await measured(
                        lambda: anthropic_probe(engine, client, body),
                        restore=lambda: anthropic_probe(
                            engine, client, body, cold=True
                        ),
                    )
                    emit(
                        {
                            "event": "request",
                            "kind": "anthropic-cold",
                            "size": 0,
                            **cold,
                        }
                    )
                    emit(
                        {
                            "event": "request",
                            "kind": "anthropic-warm",
                            "size": 0,
                            **warm,
                        }
                    )
                    validate_pair(cold, warm, a.require_hit)
                    if a.require_hit and not cold["created"]:
                        raise ValueError(
                            "image cache_control did not create a checkpoint"
                        )
                    prime_body = copy.deepcopy(body)
                    body["messages"] += [
                        {"role": "assistant", "content": cold["text"]},
                        {
                            "role": "user",
                            "content": "What colour was the image? One word.",
                        },
                    ]
                    hit = await measured(
                        lambda: anthropic_probe(engine, client, body),
                        restore=lambda: anthropic_probe(
                            engine, client, prime_body, cold=True
                        ),
                    )
                    miss = await measured(
                        lambda: anthropic_probe(engine, client, body, cold=True)
                    )
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
            scores = {"cold": 0, "hit": 0}
            for i in range(a.parity_items):
                msg, gold = arithmetic_item(i)
                if a.image_audio:
                    msg = with_audio(msg)
                cold = await probe(engine, msg, cold=True)
                # These items test hit == cold output, so the hit must restore: on a ~171-token
                # item the media restore-cost model skips the restore (it charges the whole cold
                # latency as launch cost) and the hit request had cached=0 (E1 run 5, 11-09).
                cost = getattr(engine._apc_backend, "media_restore_cost", None)
                saved = cost.cold_s if cost is not None else None
                if cost is not None:
                    cost.cold_s = (
                        0.0  # worth() admits every restore while cold_s is unset
                    )
                try:
                    hit = await probe(engine, msg)
                finally:
                    if cost is not None:
                        cost.cold_s = saved
                emit(
                    {
                        "event": "parity_item",
                        "i": i,
                        "gold": gold,
                        "cold_ids": cold["ids"],
                        "hit_ids": hit["ids"],
                        "cached": hit["cached"],
                        "restore_skipped": hit["restore_skipped"],
                        "pt": hit["pt"],
                        "cold_correct": cold["text"].strip() == str(gold),
                        "hit_correct": hit["text"].strip() == str(gold),
                    }
                )
                validate_pair(cold, hit, True)
                scores["cold"] += cold["text"].strip() == str(gold)
                scores["hit"] += hit["text"].strip() == str(gold)
            if a.parity_items:
                if abs(scores["cold"] - scores["hit"]) > 1:
                    raise ValueError("paired quality changed by more than one answer")
                emit({"event": "quality", "n": a.parity_items, "scores": scores})
            emit({"complete": True})
    finally:
        set_engine(None)
        await engine.stop()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--sizes", type=int, nargs="+", default=[1, 32768]
    )  # <= overhead+8: minimal request, label = served
    p.add_argument("--parity-items", type=int, default=0)
    p.add_argument("--require-hit", action="store_true")
    p.add_argument("--image-audio", action="store_true")
    p.add_argument(
        "--skip-anthropic",
        action="store_true",
        help="baseline arm: compare engine sessions; API cold/hit controls run on candidate",
    )
    return p


if __name__ == "__main__":
    ap = parser()
    args = ap.parse_args()
    if args.require_hit and args.skip_anthropic:
        ap.error("candidate must exercise Anthropic cache_control")
    asyncio.run(run(args))

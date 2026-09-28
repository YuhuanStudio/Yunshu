"""Reproduce backend contract differences without loading model weights.

Run from repo root: PYTHONPATH=python uv run python scripts/research/probe_backend_contracts.py
These are observations of current behavior, not a production-readiness test.
"""

import asyncio
import contextlib
import json
import platform
import threading
import time

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from yunshu_engine.batched_engine import BatchedEngine
from yunshu_engine.kv_prefix_cache import KVPrefixCache
from yunshu_engine.omni_engine import OmniEngine


async def main():
    rows = {}
    cache = KVPrefixCache()
    layer = KVCache()
    layer.update_and_fetch(mx.zeros((1, 1, 100, 4)), mx.zeros((1, 1, 100, 4)))
    cache.add(mx.array(list(range(100))), [layer])
    _, remaining, matched = cache.get(mx.array(list(range(110))))
    rows["standard_prefix_tail"] = dict(
        stored=100, query=110, matched=matched, remaining=remaining
    )
    assert matched == 100 and remaining == 10

    from yunshu_engine.batched_engine import _build_constrained_sampler

    def sampler(x):
        return x

    constrained = _build_constrained_sampler(
        sampler, {"type": "regex", "pattern": "["}, None
    )
    rows["invalid_regex"] = dict(returned_unconstrained_sampler=constrained is sampler)
    assert constrained is sampler

    import mlx.nn as nn

    from yunshu_engine.mlxvlm_mtp import _tolerant_target_load

    incomplete = nn.Linear(2, 2)
    rejected = False
    try:
        incomplete.load_weights([])
    except ValueError:
        rejected = True
    with _tolerant_target_load():
        incomplete.load_weights([])
    rows["mtp_missing_weights"] = dict(strict_rejected=rejected, tolerant_accepted=True)

    from unittest.mock import patch

    from yunshu_engine.vlm_engine import VLMEngine

    video_engine = object.__new__(VLMEngine)
    video_engine._register_temp_file = lambda _: None
    with patch("subprocess.run", side_effect=FileNotFoundError("probe: no ffmpeg")):
        frames = await video_engine._extract_frames_from_file("probe-video.mp4")
    rows["video_decode_failure"] = dict(returned_frames=frames, raised=False)

    class Backend:
        def generate(self, messages, **kwargs):
            self.kwargs = kwargs
            time.sleep(0.08)
            self.finished = True
            return dict(
                text="hello EARLY middle LATE end", completion_tokens=8, prompt_tokens=3
            )

    backend = Backend()
    engine = object.__new__(BatchedEngine)
    engine._mlxvlm_mtp = backend
    engine._apply_chat_template = lambda *_: "prompt"
    started = time.monotonic()
    chunks = []
    async for chunk in engine.stream_chat(
        [],
        max_tokens=8,
        stop=["LATE", "EARLY"],
        json_schema={"type": "object"},
        top_p=0.2,
    ):
        chunks.append(chunk)
        first_s = time.monotonic() - started
    rows["mtp_stream"] = dict(
        chunks=len(chunks),
        first_s=first_s,
        finished_before_first=backend.finished,
        forwarded_parameters=sorted(backend.kwargs),
        output=chunks[0].text,
        finish_reason=chunks[0].finish_reason,
        completion_tokens=chunks[0].completion_tokens,
    )
    assert len(chunks) == 1 and "json_schema" not in backend.kwargs
    assert chunks[0].text == "hello EARLY middle "

    entered, release, ended = threading.Event(), threading.Event(), threading.Event()

    def blocking_generate(*args, **kwargs):
        entered.set()
        release.wait(2)
        ended.set()
        return dict(text="done", completion_tokens=1, prompt_tokens=1)

    backend.generate = blocking_generate
    task = asyncio.create_task(engine.chat([]))
    while not entered.is_set():
        await asyncio.sleep(0.001)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    rows["mtp_cancel"] = dict(
        awaiter_cancelled=task.cancelled(), worker_still_running=not ended.is_set()
    )
    release.set()
    while not ended.is_set():
        await asyncio.sleep(0.001)

    # Real tokenizer; no model tensors loaded. A byte-split Unicode sequence is
    # decoded by the actual OmniEngine fragment method one token at a time.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "models/Qwen3-Omni-30B-A3B-Instruct-4bit", local_files_only=True
    )
    text = "🦄繁體中文𠀋"
    ids = tokenizer.encode(text, add_special_tokens=False)
    omni = object.__new__(OmniEngine)
    omni.processor = tokenizer
    omni._prev_text_ids = []
    fragments = [omni._decode_fragment(ids[:i]) for i in range(1, len(ids) + 1)]
    rows["omni_unicode"] = dict(
        input=text,
        ids=ids,
        whole=tokenizer.decode(ids),
        fragments=fragments,
        reconstructed="".join(fragments),
    )
    print(
        json.dumps(
            dict(
                environment=dict(
                    python=platform.python_version(),
                    mlx=mx.__version__,
                    machine=platform.machine(),
                ),
                observations=rows,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())

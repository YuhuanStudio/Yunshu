"""Reproduce backend contract differences without loading model weights.

Run from repo root: PYTHONPATH=python uv run python scripts/research/probe_backend_contracts.py
These are observations of current behavior, not a production-readiness test.
"""

import asyncio
import json
import platform

import mlx.core as mx
from mlx_lm.models.cache import KVCache

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

#!/usr/bin/env python3
"""Audit the public ordinary-LM draft route, exact greedy IDs and interleaved wall time.

Use only through gpuq for actual model runs. --dry-run validates config, tokenizer
compatibility, imports and argument plumbing without loading weights or using MLX.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

PROMPTS = (
    "Explain why a Python context manager releases resources on exceptions.",
    "Write a Python function that removes duplicates while preserving order.",
    "Give a detailed account of how a compiler parses an arithmetic expression.",
)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--draft", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def preflight(target, draft):
    from transformers import AutoTokenizer

    for path in (target, draft):
        config = json.loads((path / "config.json").read_text())
        if config.get("vision_config") is not None:
            raise ValueError(
                "external draft probe requires text-only mlx-lm checkpoints"
            )
    a = AutoTokenizer.from_pretrained(target, local_files_only=True)
    b = AutoTokenizer.from_pretrained(draft, local_files_only=True)
    if a.get_vocab() != b.get_vocab() or a.eos_token_id != b.eos_token_id:
        raise ValueError("target/draft vocabularies differ; pair is not safe")
    return a


def digest(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


async def measure(args):
    from yunshu_engine.batched_engine import BatchedEngine

    environment = {
        "YUNSHU_SPEC_UNVERIFIED": "eagle",
        "YUNSHU_DRAFT_MODEL": str(args.draft),
        "YUNSHU_ENGINE_LOOP": "0",
        "YUNSHU_NGRAM_DEFAULT": "0",
        "YUNSHU_OVERLAP": "",
        "YUNSHU_SSD_CACHE": "0",
        "YUNSHU_GPU_SAMPLER": "0",
    }
    rows = []
    with patch.dict(os.environ, environment):
        engine = BatchedEngine(model_name=str(args.target))
        await engine.start()
        try:
            if not engine._spec_enabled or engine._spec_decoder is None:
                raise RuntimeError("public external draft route did not initialize")
            mode = engine._spec_route(
                spec_decode=True,
                stream=False,
                temperature=0.0,
                logprobs=False,
                use_engine_loop=False,
            )
            if mode != "eagle":
                raise RuntimeError(f"wrong engaged route: {mode}")
            await engine.generate(
                "Say hello.",
                max_tokens=8,
                temperature=0.0,
                spec_decode=False,
                use_engine_loop=False,
            )
            await engine.generate(
                "Say hello.",
                max_tokens=8,
                temperature=0.0,
                spec_decode=True,
                use_engine_loop=False,
            )
            generation = importlib.import_module("mlx_lm.generate")
            plain_generate = generation.generate_step
            draft_generate = engine._spec_decoder.generate
            observed = []

            def trace_plain(*a, **kw):
                for token, lp in plain_generate(*a, **kw):
                    observed.append(int(token.item()))
                    yield token, lp

            def trace_draft(*a, **kw):
                ids = draft_generate(*a, **kw)
                observed.extend(ids)
                return ids

            eos = engine._tokenizer.eos_token_id
            eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
            repeats = 1 if args.smoke else args.repeats
            tokens = min(args.tokens, 16) if args.smoke else args.tokens
            for rep in range(repeats):
                order = ["plain", "external"] if rep % 2 == 0 else ["external", "plain"]
                for i, prompt in enumerate(PROMPTS[:1] if args.smoke else PROMPTS):
                    pair = {}
                    for arm in order:
                        observed.clear()
                        with (
                            patch.object(generation, "generate_step", trace_plain),
                            patch.object(engine._spec_decoder, "generate", trace_draft),
                        ):
                            result = await engine.generate(
                                prompt,
                                max_tokens=tokens,
                                temperature=0.0,
                                spec_decode=arm == "external",
                                use_engine_loop=False,
                                enable_thinking=False,
                            )
                        ids = list(observed)
                        stop = next(
                            (i for i, token in enumerate(ids) if token in eos), len(ids)
                        )
                        ids = ids[:stop][:tokens]
                        if not ids or result.finish_reason not in ("stop", "length"):
                            raise RuntimeError(
                                f"incomplete generation: {result.finish_reason}"
                            )
                        pair[arm] = ids
                        # Timing calls exclude trace instrumentation entirely.
                        start = time.perf_counter()
                        timed = await engine.generate(
                            prompt,
                            max_tokens=tokens,
                            temperature=0.0,
                            spec_decode=arm == "external",
                            use_engine_loop=False,
                            enable_thinking=False,
                        )
                        elapsed = time.perf_counter() - start
                        if (
                            timed.text != result.text
                            or timed.finish_reason != result.finish_reason
                        ):
                            raise RuntimeError("trace/timing call output changed")
                        row = {
                            "rep": rep,
                            "prompt": i,
                            "arm": arm,
                            "engaged_mode": "eagle" if arm == "external" else "fast",
                            "token_ids": ids,
                            "digest": digest(ids),
                            "wall_s": elapsed,
                            "completion_tokens": timed.completion_tokens,
                            "wall_tps": timed.completion_tokens / elapsed,
                        }
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                    if pair["plain"] != pair["external"]:
                        raise RuntimeError(
                            f"greedy token parity failed: rep={rep} prompt={i}"
                        )
        finally:
            await engine.stop()
    return rows


def main():
    ap = parser()
    args = ap.parse_args()
    if args.repeats < 3 and not args.smoke:
        ap.error("at least three interleaved runs required")
    preflight(args.target, args.draft)
    from yunshu_engine.batched_engine import BatchedEngine  # validates serving imports

    if not hasattr(BatchedEngine, "_init_external_lm_spec"):
        raise RuntimeError("external route absent from selected source")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "target": str(args.target),
        "draft": str(args.draft),
        "load_start": os.getloadavg(),
        "complete": False,
    }
    try:
        result["rows"] = [] if args.dry_run else asyncio.run(measure(args))
        result["complete"] = "dry-run" if args.dry_run else True
        result["load_end"] = os.getloadavg()
    finally:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()

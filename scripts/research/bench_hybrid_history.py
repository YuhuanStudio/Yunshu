"""Real Qwen3.8 VLM text history probe using an existing local checkpoint.

Run in separate processes with YUNSHU_VLM_HYBRID_PREFIX and
YUNSHU_VLM_HYBRID_PREFIX_BLOCK set explicitly. No model is downloaded.
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path


async def run(args):
    import mlx.core as mx

    from yunshu_engine.vlm_engine import VLMEngine

    engine = VLMEngine(str(args.model))
    await engine.start()
    try:
        history = [
            {
                "role": "user",
                "content": ("The archive record is neutral background text.\n" * 300)
                + "\nThe current code is ALPHA. Remember it.",
            },
            {"role": "assistant", "content": "ALPHA"},
        ]
        for turn in range(args.turns):
            expected = "ALPHA" if turn < args.change_turn else "COBALT"
            if turn == args.change_turn:
                question = "The current code changed to COBALT. What is the current code? Reply only with the code."
            else:
                question = "What is the current code? Reply only with the code."
            messages = history + [{"role": "user", "content": question}]
            mx.reset_peak_memory()
            started = time.perf_counter()
            first = None
            pieces = []
            last = None
            try:
                async for item in engine.generate_stream(
                    messages=messages,
                    max_tokens=24,
                    temperature=0,
                    enable_thinking=False,
                ):
                    last = item
                    if item.new_text and first is None:
                        first = time.perf_counter() - started
                    pieces.append(item.new_text)
                output = "".join(pieces).strip()
                row = {
                    "turn": turn,
                    "expected": expected,
                    "output": output,
                    "task_ok": output == expected,
                    "first_text_s": first,
                    "wall_s": time.perf_counter() - started,
                    "prompt_tokens": last.prompt_tokens if last else None,
                    "cached_tokens": last.cached_tokens if last else None,
                    "finish_reason": last.finish_reason if last else None,
                    "peak_bytes": mx.get_peak_memory(),
                    "active_bytes": mx.get_active_memory(),
                    "cache_stats": engine._text_kv_prefix_cache.get_stats(),
                }
                history.extend(
                    [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": output},
                    ]
                )
            except Exception as exc:
                row = {"turn": turn, "error": repr(exc), "task_ok": False}
            with args.output.open("a") as file:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                {
                    k: row.get(k)
                    for k in (
                        "turn",
                        "task_ok",
                        "first_text_s",
                        "cached_tokens",
                        "output",
                        "error",
                    )
                    if k in row
                },
                flush=True,
            )
    finally:
        await engine.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--change-turn", type=int, default=10)
    args = parser.parse_args()
    args.model = args.model.expanduser().resolve()
    if (
        not args.model.is_relative_to(Path("/Volumes/P5Plus"))
        or not (args.model / "config.json").is_file()
    ):
        parser.error("Use an existing model directory on /Volumes/P5Plus")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
